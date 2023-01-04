from ray.tune.integration.pytorch_lightning import TuneReportCallback
from .config import Config, DataBatch
from .datamodule import DataModule
from .metamodel import Metamodel
import pytorch_lightning as ptl


def train(conf: Config):
    """
    Performis training, given a set of hyperparameters
    as a `Config` typed dict.
    """
    ptl.seed_everything(conf["random_seed"], workers=True)
    ptl.Trainer(
        deterministic=True,
        callbacks=[
            TuneReportCallback(
                metrics={"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"},
                on="validation_end",
            )
        ],
        fast_dev_run=conf["debug"],
    ).fit(
        model=Metamodel(conf),
        datamodule=DataModule(conf),
    )
