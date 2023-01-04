import pytorch_lightning as ptl
import torch
import torchmetrics

from epo.supervised.config import DataBatch
from . import Config


class Metamodel(ptl.LightningModule):
    """A model training/evaluation template"""

    def __init__(self, conf: Config):
        super().__init__()
        self.conf = conf
        self.classifier = conf["classifier"]
        self.metrics = torch.nn.ModuleDict(
            dict(
                loss=torch.nn.CrossEntropyLoss(),
                accuracy=torchmetrics.Accuracy.__new__(
                    torchmetrics.Accuracy,
                    task="binary",
                ),
                recall=torchmetrics.Recall.__new__(
                    torchmetrics.Recall,
                    task="binary",
                ),
                f1=torchmetrics.F1Score.__new__(
                    torchmetrics.F1Score,
                    task="binary",
                ),
            )
        )

    def forward(
        self,
        tokens: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.classifier(tokens, mask).logits

    def __step(self, batch: DataBatch, prefix: str) -> torch.Tensor:
        """Generic step"""
        src, mask, tgt = batch
        res = self(src, mask)
        logs = {f"{prefix}_{n}": f(res, tgt) for n, f in self.metrics.items()}
        return logs[f"{prefix}_loss"]

    def training_step(self, batch: DataBatch, batch_idx: int):
        return self.__step(batch, "train")

    def validation_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        return self.__step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.conf["optimizer"](self)
