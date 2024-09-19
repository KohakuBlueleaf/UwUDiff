import warnings
from PIL import Image
from pathlib import Path
from collections.abc import Callable

from torchvision import transforms as T
from torch.utils.data import Dataset

from duwu.utils import get_images_recursively


class LocalImageDataset(Dataset):

    def __init__(self, image_paths: list[str], image_transform: Callable | None = None):
        self.image_paths = image_paths
        self.image_transform = image_transform or T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Custom warning handler
        def custom_showwarning(
            message, category, filename, lineno, file=None, line=None
        ):
            print(f"{image_path}: {message}")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            warnings.showwarning = custom_showwarning

            with Image.open(image_path).convert("RGB") as image:
                image = self.image_transform(image)

        return image


class LocalImageDatasetFromFolder(LocalImageDataset):

    def __init__(self, image_dir: str, image_transform: Callable | None = None):
        image_paths = get_images_recursively(image_dir)
        super().__init__(image_paths, image_transform)


class LocalTextImageDataset(LocalImageDataset):

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        image_path = self.image_paths[idx]
        txt_path = Path(image_path).with_suffix(".txt")
        with open(txt_path, "r") as f:
            text = f.read().strip()
        return image, text
