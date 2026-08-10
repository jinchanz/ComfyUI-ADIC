import comfy.utils
import torch
import torch.nn.functional as F


class ImageStitch:
    """Upstreamed from https://github.com/kijai/ComfyUI-KJNodes"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "step": 2},
                ),
                "image_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "可选图片 URL，每行一个，0 个或多个",
                    },
                ),
                "spacing_color": (
                    ["white", "black", "red", "green", "blue"],
                    {"default": "white"},
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "image/transform"
    DESCRIPTION = """
Stitches image2 to image1 in the specified direction.
If image2 is not provided, returns image1 unchanged.
Optional spacing can be added between images.
"""

    def stitch(
        self,
        image1,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        image2=None,
    ):
        if image2 is None:
            return (image1,)

        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat([image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)])
            if image2.shape[0] < max_batch:
                image2 = torch.cat([image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)])

        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2

            if direction in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:
                target_w, target_h = w1, int(w1 / aspect_ratio)

            image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled").movedim(1, -1)

        color_map = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        color_val = color_map[spacing_color]

        if not match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            pad_value = 0.0
            if not isinstance(color_val, tuple):
                pad_value = color_val

            if direction in ["left", "right"]:
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        pad_h = target_h - h1
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode="constant", value=pad_value)
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode="constant", value=pad_value)
            else:
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode="constant", value=pad_value)
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode="constant", value=pad_value)

        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1],
                            max_channels - image1.shape[-1],
                            device=image1.device,
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < max_channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1],
                            max_channels - image2.shape[-1],
                            device=image2.device,
                        ),
                    ],
                    dim=-1,
                )

        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)

            if direction in ["left", "right"]:
                spacing_shape = (
                    image1.shape[0],
                    max(image1.shape[1], image2.shape[1]),
                    spacing_width,
                    image1.shape[-1],
                )
            else:
                spacing_shape = (
                    image1.shape[0],
                    spacing_width,
                    max(image1.shape[2], image2.shape[2]),
                    image1.shape[-1],
                )

            spacing = torch.full(spacing_shape, 0.0, device=image1.device)
            if isinstance(color_val, tuple):
                for i, c in enumerate(color_val):
                    if i < spacing.shape[-1]:
                        spacing[..., i] = c
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color_val
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0

        images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        return (torch.cat(images, dim=concat_dim),)


def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c


class ReferenceLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/edit_models"
    DESCRIPTION = "This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images."

    def append(self, conditioning, latent=None):
        if latent is not None:
            conditioning = conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        return (conditioning,)


PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class FluxKontextImageScale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    CATEGORY = "advanced/conditioning/flux"
    DESCRIPTION = "This node resizes the image to one that is more optimal for flux kontext."

    def scale(self, image):
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        return (image,)


class ImageConcatFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_columns": ("INT", {"default": 3, "min": 1, "max": 255, "step": 1}),
                "match_image_size": ("BOOLEAN", {"default": False}),
                "max_resolution": ("INT", {"default": 4096}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
    Concatenates images from a batch into a grid with a specified number of columns.
    """

    def concat(self, images, num_columns, match_image_size, max_resolution):
        batch_size, height, width, channels = images.shape
        num_rows = (batch_size + num_columns - 1) // num_columns

        print(f"Initial dimensions: batch_size={batch_size}, height={height}, width={width}, channels={channels}")
        print(f"num_rows={num_rows}, num_columns={num_columns}")

        if match_image_size:
            target_shape = images[0].shape

            resized_images = []
            for image in images:
                original_height = image.shape[0]
                original_width = image.shape[1]
                original_aspect_ratio = original_width / original_height

                if original_aspect_ratio > 1:
                    target_height = target_shape[0]
                    target_width = int(target_height * original_aspect_ratio)
                else:
                    target_width = target_shape[1]
                    target_height = int(target_width / original_aspect_ratio)

                print(f"Resizing image from ({original_height}, {original_width}) to ({target_height}, {target_width})")

                image_for_resize = image.permute(2, 0, 1).unsqueeze(0)

                resized_image = F.interpolate(image_for_resize, size=(target_height, target_width), mode="bilinear", align_corners=False)

                resized_image = resized_image.squeeze(0).permute(1, 2, 0)
                resized_images.append(resized_image)

            images = torch.stack(resized_images)
            height, width = target_shape[:2]

        grid_height = num_rows * height
        grid_width = num_columns * width

        print(f"Grid dimensions before scaling: grid_height={grid_height}, grid_width={grid_width}")

        scale_factor = min(max_resolution / grid_height, max_resolution / grid_width, 1.0)

        scaled_height = height * scale_factor
        scaled_width = width * scale_factor

        height = max(8, int(round(scaled_height / 8) * 8))
        width = max(8, int(round(scaled_width / 8) * 8))

        grid_height = num_rows * height
        grid_width = num_columns * width

        print(f"Grid dimensions after scaling: grid_height={grid_height}, grid_width={grid_width}")
        print(f"Final image dimensions: height={height}, width={width}")

        grid = torch.zeros((grid_height, grid_width, channels), dtype=images.dtype, device=images.device)

        for idx, image in enumerate(images):
            if idx >= batch_size:
                break

            image_for_resize = image.permute(2, 0, 1).unsqueeze(0)

            resized_image = F.interpolate(image_for_resize, size=(height, width), mode="bilinear", align_corners=False)

            resized_image = resized_image.squeeze(0).permute(1, 2, 0)

            row = idx // num_columns
            col = idx % num_columns

            start_row = row * height
            end_row = min(start_row + height, grid_height)
            start_col = col * width
            end_col = min(start_col + width, grid_width)

            grid[start_row:end_row, start_col:end_col, :] = resized_image[: end_row - start_row, : end_col - start_col, :]

        return (grid.unsqueeze(0),)
