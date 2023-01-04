import pytorch_lightning as ptl
import torch

from epo.supervised.config import DataBatch
from . import Config


class Metamodel(ptl.LightningModule):
    """A model training/evaluation template"""

    def __init__(self, conf: Config):
        super().__init__()
        self.conf = conf
        self.classifier = conf["classifier"]

    def forward(self, tokens: torch.LongTensor, mask: torch.FloatTensor):
        return self.classifier(tokens, mask).logits

    def training_step(self, batch: DataBatch, batch_idx: int):
        print(batch.labels)
        src, mask, tgt = batch
        res = self(src, mask)

        return

    def validation_step(self, batch: DataBatch, batch_idx: int):
        return

    def configure_optimizers(self):
        return self.conf["optimizer"](self)
