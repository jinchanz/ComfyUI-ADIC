import base64
import asyncio
import io
import math
import time
from typing import Any, Optional

import requests
import torch
import numpy as np
from PIL import Image

import comfy.utils

# 与旧版 comfy_api_nodes 实现保持一致的默认值
DEFAULT_DOWNSCALE_TOTAL_PIXELS = 1536 * 1024
DEFAULT_DATA_URI_TOTAL_PIXELS = 2048 * 2048
DEFAULT_DOWNLOAD_MAX_RETRIES = 3
DEFAULT_DOWNLOAD_RETRY_DELAY = 1.0
DEFAULT_DOWNLOAD_RETRY_BACKOFF = 2.0

MIME_TYPE_TO_PIL_FORMAT = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/webp": "WEBP",
}


def validate_string(
    string: str,
    strip_whitespace: bool = True,
    field_name: str = "prompt",
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> None:
    """与 comfy_api_nodes 的 validate_string 行为一致：空串本身合法，只在超出长度约束时报错。"""
    if string is None:
        raise ValueError(f"字段 '{field_name}' 不能为 None")
    if strip_whitespace:
        string = string.strip()
    if min_length and len(string) < min_length:
        raise ValueError(f"字段 '{field_name}' 长度不能小于 {min_length}，当前为 {len(string)}")
    if max_length and len(string) > max_length:
        raise ValueError(f"字段 '{field_name}' 长度不能大于 {max_length}，当前为 {len(string)}")


def downscale_image_tensor(image: torch.Tensor, total_pixels: int = DEFAULT_DOWNSCALE_TOTAL_PIXELS) -> torch.Tensor:
    """按总像素数等比缩小图片（lanczos），与旧版 comfy 实现一致。"""
    if not isinstance(image, torch.Tensor):
        raise ValueError("image 必须是 torch.Tensor")
    if image.dim() != 4:
        raise ValueError("image 形状必须为 (B, H, W, C)")

    samples = image.movedim(-1, 1)
    scale_by = math.sqrt(int(total_pixels) / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image

    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    scaled = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return scaled.movedim(1, -1)


async def download_url_to_bytesio(url: str, timeout: Optional[float] = None) -> io.BytesIO:
    """下载图片，带指数退避重试（对齐旧版 client 的重试策略）。"""

    def _download() -> io.BytesIO:
        delay = DEFAULT_DOWNLOAD_RETRY_DELAY
        last_error: Optional[Exception] = None

        for attempt in range(DEFAULT_DOWNLOAD_MAX_RETRIES):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return io.BytesIO(response.content)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                # 4xx 属于确定性失败，重试没有意义
                if status is not None and 400 <= status < 500:
                    raise
                last_error = exc
            except requests.RequestException as exc:
                last_error = exc

            if attempt < DEFAULT_DOWNLOAD_MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= DEFAULT_DOWNLOAD_RETRY_BACKOFF

        raise last_error if last_error else RuntimeError(f"下载图片失败: {url}")

    return await asyncio.to_thread(_download)


def tensor_to_data_uri(
    image_tensor: torch.Tensor,
    total_pixels: Optional[int] = DEFAULT_DATA_URI_TOTAL_PIXELS,
    mime_type: str = "image/png",
) -> str:
    if image_tensor.dim() == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError("image_tensor 形状必须为 (H, W, C) 或 (1, H, W, C)")
        image_tensor = image_tensor[0]
    if image_tensor.dim() != 3:
        raise ValueError("image_tensor 形状必须为 (H, W, C)")

    if total_pixels:
        image_tensor = downscale_image_tensor(image_tensor.unsqueeze(0), total_pixels)[0]

    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image_np = (image.numpy() * 255.0).astype("uint8")

    pil_image = Image.fromarray(image_np)
    image_format = MIME_TYPE_TO_PIL_FORMAT.get(mime_type.lower(), "PNG")

    # JPEG 不支持 alpha 通道
    if image_format == "JPEG" and pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")

    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format=image_format)
    encoded = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def bytesio_to_image_tensor(data: io.BytesIO, mode: str = "RGBA") -> torch.Tensor:
    """默认输出 RGBA（4 通道），与旧版 comfy 实现一致，保留透明背景。"""
    data.seek(0)
    image = Image.open(data).convert(mode)
    tensor = torch.from_numpy(np.array(image).astype("float32") / 255.0)
    return tensor.unsqueeze(0)


def _send_progress_text(text: str, node_id: Any) -> None:
    if not node_id:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        pass


async def validate_and_cast_response(
    response: Any,
    timeout: Optional[float] = None,
    node_id: Any = None,
) -> torch.Tensor:
    """把 API 响应中的全部图片转成 (B, H, W, C) 张量，与旧版 comfy 实现一致。"""
    data_items = getattr(response, "data", None)
    if data_items is None and isinstance(response, dict):
        data_items = response.get("data")
    if not data_items:
        raise ValueError("API 响应中缺少图片数据")

    image_tensors: list[torch.Tensor] = []

    for image_item in data_items:
        if isinstance(image_item, dict):
            image_url = image_item.get("url")
            image_b64 = image_item.get("b64_json")
        else:
            image_url = getattr(image_item, "url", None)
            image_b64 = getattr(image_item, "b64_json", None)

        if image_b64:
            buffer = io.BytesIO(base64.b64decode(image_b64))
        elif image_url:
            if image_url.startswith("data:"):
                _, _, encoded = image_url.partition(",")
                buffer = io.BytesIO(base64.b64decode(encoded))
            else:
                _send_progress_text(f"Result URL: {image_url}", node_id)
                buffer = await download_url_to_bytesio(image_url, timeout=timeout)
        else:
            raise ValueError("API 未返回可用的图片内容")

        image_tensors.append(bytesio_to_image_tensor(buffer)[0])

    return torch.stack(image_tensors, dim=0)
