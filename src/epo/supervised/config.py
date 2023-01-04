from typing import Callable, NamedTuple, TypedDict
import torch


class DataBatch(NamedTuple):
    """
    A tuple, which consists of:
    1. Document claim tensor
    2. Document claim mask
    2. Document class tensor
    """

    tokens: torch.LongTensor
    attention_masks: torch.FloatTensor
    labels: torch.FloatTensor


class Config(TypedDict):
    """A set of parameters, which are used to set the single run up."""

    # ---- GENRAL TRAINING PARAMETERS ---- #
    random_seed: int
    train_val_split: float
    data_dir: str
    batch_size: int
    num_workers: int
    debug: bool

    # ---- NEURAL NETWORK ---- #
    # The neural network consists of the following parts:
    # 1. Tokenizer -- some function, which turns raw data into a tensor
    #    that can later be passed into the encoder
    tokenizer: Callable[[tuple[bool, list[str]]], DataBatch]
    # 2. Classifier -- a model, which learns to map token space tensors into the
    #    class tensors
    classifier: torch.nn.Module
    # 3. Optimizer
    optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer]
