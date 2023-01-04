from typing import Any, List, Tuple
from .config import DataBatch
from . import Config
from multiprocessing import Pool
import pytorch_lightning as ptl
import torch.utils.data
import json
from itertools import chain
from tqdm.auto import tqdm


class DataModule(ptl.LightningDataModule):
    def __init__(self, conf: Config):
        super().__init__()
        self.conf = conf

    def collate_batch(self, batch: List[DataBatch]) -> DataBatch:
        """Custom batch creator"""
        tokens = map(lambda x: x.tokens, batch)
        masks = map(lambda x: x.attention_masks, batch)
        label = map(lambda x: x.labels, batch)
        return DataBatch(
            tokens=torch.LongTensor(torch.stack(list(tokens))),
            attention_masks=torch.FloatTensor(torch.stack(list(masks))),
            labels=torch.FloatTensor(torch.stack(list(label))),
        )

    def setup(self, stage: str) -> None:
        # Loading datasets
        with open(f"{self.conf['data_dir']}/dsY02W.json") as file:
            json_y02w = json.load(file)
        with open(f"{self.conf['data_dir']}/dsOTHER.json") as file:
            json_other = json.load(file)

        if self.conf["debug"]:
            json_y02w = json_y02w[:5]
            json_other = json_other[:5]

        # Setting classes for the datasets
        total_len = sum(map(len, (json_other, json_y02w)))
        class_claims: chain[Tuple[bool, List[str]]] = chain(
            *[
                [(cls, claims) for _, claims in data]
                for cls, data in [
                    (True, json_y02w),
                    (False, json_other),
                ]
            ]
        )

        # Performing tokenization in a thread pool
        # with Pool() as pool:
        #     dataset = list(
        #         tqdm(
        #             pool.imap(
        #                 self.conf["tokenizer"],
        #                 class_claims,
        #             ),
        #             total=total_len,
        #             desc="Tokenizing `Y02W vs OTHER` data",
        #         )
        #     )

        dataset = [
            self.conf["tokenizer"](c)
            for c in tqdm(class_claims, desc="Tokenizing `Y02W vs OTHER` data")
        ]
        train_val_split_index = int(len(dataset) * self.conf["train_val_split"])
        self.train_dataset, self.val_dataset = (
            dataset[:train_val_split_index],
            dataset[train_val_split_index:],
        )

    def __dataloader(self, dataset: Any) -> torch.utils.data.DataLoader[DataBatch]:
        """Generic dataloader constructor"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.conf["batch_size"],
            num_workers=self.conf["num_workers"],
            collate_fn=self.collate_batch,
        )

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset)
