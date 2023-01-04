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
        gpus=conf["gpus"],
        deterministic=True,
        callbacks=[
            TuneReportCallback(
                metrics={
                    "val_loss": "val_loss",
                    "val_accuracy": "val_accuracy",
                },
                on="validation_end",
            ),
        ],
        fast_dev_run=conf["debug"],
    ).fit(
        model=Metamodel(conf),
        datamodule=DataModule(conf),
    )
