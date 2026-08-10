import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from pydantic import BaseModel, ConfigDict, Field

from .local_api_client import ApiEndpoint, HttpMethod, SynchronousOperation
from .local_api_models import OpenAIImageEditRequest, OpenAIImageGenerationRequest, OpenAIImageGenerationResponse
from .local_api_utils import (
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    downscale_image_tensor,
    tensor_to_data_uri,
    validate_and_cast_response,
    validate_string,
)


class ADICOpenAIGPTImage1(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's GPT Image 1 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "api_base": ("STRING", {"default": ""}),
                "auth_token": ("STRING", {"default": ""}),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["low", "medium", "high"],
                        "default": "low",
                        "tooltip": "Image quality, affects cost and generation time.",
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": "Return image with or without background",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                        "default": "auto",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
            },
            "hidden": {
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    API_NODE = True

    async def api_call(
        self,
        prompt,
        seed=0,
        quality="low",
        background="opaque",
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "gpt-image-1"
        path = "/v1/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        files = []

        if image is not None:
            path = "/v1/images/edits"
            request_class = OpenAIImageEditRequest
            content_type = "multipart/form-data"

            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_image_tensor(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                if batch_size == 1:
                    files.append(("image", (f"image_{i}.png", img_byte_arr, "image/png")))
                else:
                    files.append(("image[]", (f"image_{i}.png", img_byte_arr, "image/png")))

        if mask is not None:
            if image is None:
                raise Exception("Cannot use a mask without an input image")
            if image.shape[0] != 1:
                raise Exception("Cannot use a mask with multiple image")
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

            scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0)).squeeze()

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

        api_base = ""
        if kwargs is not None:
            api_base = kwargs.get("api_base")

        operation = SynchronousOperation(
            api_base=api_base or "https://api.gpt.ge",
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                quality=quality,
                background=background,
                n=n,
                seed=seed,
                size=size,
            ),
            files=files if files else None,
            content_type=content_type,
            auth_kwargs=kwargs,
        )

        response = await operation.execute()

        img_tensor = await validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class GeminiChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False


class GeminiChatCompletionChoice(BaseModel):
    message: Dict[str, Any]


class GeminiChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: Optional[bool] = None
    message: Optional[str] = None
    request_id: Optional[str] = Field(default=None, alias="requestId")
    choices: Optional[List[GeminiChatCompletionChoice]] = None


class IdeaLabImageGenerate(ComfyNodeABC):
    """
    Generates images synchronously via IdeaLab Image Generate endpoint.

    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "可选参考图，支持多张图片批量输入",
                    },
                ),
                "image_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "可选图片 URL，每行一个，0 个或多个",
                    },
                ),
                "image_mime_type": (
                    "STRING",
                    {
                        "default": "image/png",
                        "tooltip": "上传到 Gemini 的图片编码格式",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gemini-3-pro-image-preview",
                        "tooltip": "可切换的 Gemini 模型名称",
                    },
                ),
                "api_base": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "API 基础地址，例如 https://host/api/openai",
                    },
                ),
                "auth_token": ("STRING", {"default": "", "tooltip": "Bearer Token"}),
            },
            "hidden": {
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/Gemini"
    API_NODE = True

    async def api_call(
        self,
        prompt,
        image=None,
        image_urls="",
        image_mime_type="image/png",
        model="gemini-3-pro-image-preview",
        api_base="",
        auth_token="",
        unique_id=None,
        **kwargs,
    ):
        content_blocks: List[Dict[str, Any]] = []
        if prompt and prompt.strip():
            validate_string(prompt, strip_whitespace=False)
            content_blocks.append({"type": "text", "text": prompt})

        if image is not None:
            if len(image.shape) < 4 or image.shape[0] == 0:
                raise ValueError("输入图片格式不正确")
            for idx in range(image.shape[0]):
                data_uri = tensor_to_data_uri(image[idx], mime_type=image_mime_type)
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

        if image_urls:
            urls = [line.strip() for line in image_urls.splitlines() if line.strip()]
            for url in urls:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

        if not content_blocks:
            raise ValueError("请至少提供文本或图片中的一种输入")

        path = "/v1/chat/completions"
        request = GeminiChatCompletionRequest(
            model=model or "gemini-3-pro-image-preview",
            messages=[
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
            stream=False,
        )
        auth_kwargs = dict(kwargs) if kwargs else {}
        if auth_token:
            auth_kwargs["auth_token"] = auth_token

        operation = SynchronousOperation(
            api_base=api_base,
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=GeminiChatCompletionRequest,
                response_model=GeminiChatCompletionResponse,
            ),
            request=request,
            auth_kwargs=auth_kwargs,
        )
        response = await operation.execute()

        if response.success is False:
            raise ValueError(response.message or "Gemini 接口返回失败")

        if not response.choices:
            raise ValueError("Gemini 接口未返回可用的结果")

        message = response.choices[0].message or {}
        content = message.get("content")
        image_url = None

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url")
                    if image_url:
                        break
        elif isinstance(content, dict):
            if content.get("type") == "image_url":
                image_url = content.get("image_url", {}).get("url")
        elif isinstance(content, str):
            image_url = content

        if not image_url:
            raise ValueError("Gemini 接口未返回图片内容")

        if image_url.startswith("http://") or image_url.startswith("https://"):
            img_bytesio = await download_url_to_bytesio(image_url)
        else:
            if image_url.startswith("data:"):
                _, _, encoded = image_url.partition(",")
            else:
                encoded = image_url
            try:
                img_bytes = base64.b64decode(encoded)
            except Exception as exc:
                raise ValueError("返回的图片内容不是有效的 Base64 数据") from exc
            img_bytesio = io.BytesIO(img_bytes)

        img_tensor = bytesio_to_image_tensor(img_bytesio)
        return (img_tensor,)
