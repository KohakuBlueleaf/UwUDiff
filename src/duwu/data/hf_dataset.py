from collections.abc import Callable

import datasets

import torch.utils.data as Data
from torchvision import transforms as T


class HfImageDataset(Data.Dataset):

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        image_key: str = "image",
        image_transform: Callable | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.image_key = image_key
        self.image_transform = image_transform or T.ToTensor()

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx][self.image_key].convert("RGB")
        image_tensor = self.image_transform(image)
        return image_tensor


class HfPromptDataset(Data.Dataset):

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        prompt_key: str = "caption",
        all_captions: bool = False,
    ):
        self.captions = []
        for sample in hf_dataset:
            if all_captions:
                self.captions.extend(sample[prompt_key])
            else:
                self.captions.append(sample[prompt_key][0])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]
