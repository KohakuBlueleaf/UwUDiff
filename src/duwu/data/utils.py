import math
from functools import partial
from PIL import Image

import numpy as np
import torch
from torchvision import transforms as T


def vae_image_postprocess(image_tensor: torch.Tensor) -> Image.Image:
    image = Image.fromarray(
        ((image_tensor * 0.5 + 0.5) * 255)
        .cpu()
        .clamp(0, 255)
        .numpy()
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    return image


BicubicResize = partial(T.Resize, interpolation=T.InterpolationMode.BICUBIC)


def resize_and_crop_image(
    image: Image.Image,
    target_size: tuple[int, int] = (256, 256),
    random_crop: bool = True,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:

    # Calculate the resize dimensions while maintaining aspect ratio
    scale_w = target_size[0] / image.width
    scale_h = target_size[1] / image.height
    scale = max(scale_w, scale_h)

    # Use math.ceil to ensure the new size is large enough
    new_size = (math.ceil(image.width * scale), math.ceil(image.height * scale))

    # Resize the image
    image = image.resize(new_size, Image.LANCZOS)
    image = T.ToTensor()(image)

    # Calculate cropping coordinates
    crop_y = new_size[1] - target_size[1]
    crop_x = new_size[0] - target_size[0]

    if random_crop:
        top = np.random.randint(0, crop_y + 1)
        left = np.random.randint(0, crop_x + 1)
    else:
        top = crop_y // 2
        left = crop_x // 2

    # Perform the crop centered around the middle of the image
    cropped_image = image[:, top : top + target_size[1], left : left + target_size[0]]
    cropped_image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cropped_image)

    return cropped_image, new_size, (left, top)
