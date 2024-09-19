import torch
import torch.utils.data as Data
from lightning import LightningDataModule
from transformers import PreTrainedTokenizer

from duwu.loader import load_any


class UwUBaseDataset(Data.Dataset):

    @staticmethod
    def collate(batch):
        samples = torch.stack([x["sample"] for x in batch])
        caption = [x["caption"] for x in batch]
        tokenizer_outs = [x["tokenizer_out"] for x in batch]
        add_time_ids = torch.stack([x["add_time_ids"] for x in batch]).float()
        tokenizer_outputs = []
        for tokenizer_out in zip(*tokenizer_outs):
            input_ids = torch.concat([x["input_ids"] for x in tokenizer_out])
            attention_mask = torch.concat([x["attention_mask"] for x in tokenizer_out])
            tokenizer_outputs.append(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        return (
            samples,
            caption,
            tokenizer_outputs,
            {"time_ids": add_time_ids},
            # cross_attention_kwargs
            {},
        )


class DummyDataset(UwUBaseDataset):
    def __init__(
        self,
        # (4, 128, 128) for latent
        sample_size: tuple[int] = (3, 1024, 1024),
        n_samples: int = 100,
        tokenizers: list[PreTrainedTokenizer] = [],
        **kwargs,
    ):
        if not isinstance(sample_size, tuple):
            sample_size = tuple(sample_size)
        self.samples = [torch.randn(sample_size) for _ in range(n_samples)]
        if isinstance(tokenizers, list):
            self.tokenizers = tokenizers
        else:
            self.tokenizers = [tokenizers]

    def set_tokenizers(self, tokenizers):
        self.tokenizers = tokenizers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        caption = "DUMMY TEST"
        return {
            "sample": sample,
            "caption": caption,
            "tokenizer_out": [
                tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                for tokenizer in self.tokenizers
            ],
            # org_h, org_w, crop_top, crop_left, target_h, target_w
            "add_time_ids": torch.tensor([1024, 1024, 0, 0, 1024, 1024]),
        }


class TrainDataModule(LightningDataModule):

    def __init__(self, dataset_config, dataloader_config):
        super().__init__()
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config

    def setup(self, stage: str):
        self.dataset = load_any(self.dataset_config)
        if hasattr(self, "tokenizers"):
            self.dataset.set_tokenizers(self.tokenizers)

    def train_dataloader(self):
        return Data.DataLoader(
            self.dataset, collate_fn=self.dataset.collate, **self.dataloader_config
        )

    def set_tokenizers(self, tokenizers):
        self.tokenizers = tokenizers
