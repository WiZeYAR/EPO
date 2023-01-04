from typing import List, Tuple
import torch
from epo.supervised import Config, DataBatch, train
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import ray.tune as tune


def prepare_batch(inp: Tuple[bool, List[str]]) -> DataBatch:
    """A function to prepare a data batch from raw data"""
    is_y02w, claims = inp[0], "".join(inp[1])
    encoding = AutoTokenizer.from_pretrained("distilbert-base-uncased")(
        claims,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    return DataBatch(
        tokens=torch.LongTensor(encoding.input_ids),
        attention_masks=torch.FloatTensor(encoding.attention_mask),
        labels=torch.FloatTensor([1, 0] if is_y02w else [0, 1]),
    )


def train_single():
    train(
        Config(
            num_workers=1,
            debug=True,
            random_seed=0,
            data_dir="./data",
            train_val_split=0.8,
            batch_size=5,
            tokenizer=prepare_batch,
            gpus=1 if torch.cuda.is_available() else 0,
            classifier=DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            ),
            optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        )
    )


def train_tune():
    search_space = dict(
        num_workers=1,
        debug=True,
        random_seed=0,
        data_dir="/home/wize/Repositories/EPO/data",
        train_val_split=0.8,
        batch_size=tune.choice([4, 8]),
        tokenizer=prepare_batch,
        gpus=0,
        classifier=DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        ),
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    )
    experiment = tune.run(train, config=search_space, num_samples=2)
    print(experiment.results)


if __name__ == "__main__":
    train_tune()
