import torch
from epo.supervised import Config, DataBatch, train
from transformers import AutoTokenizer, DistilBertForSequenceClassification


def prepare_batch(inp: tuple[bool, list[str]]) -> DataBatch:
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


if __name__ == "__main__":
    train(
        Config(
            num_workers=8,
            debug=True,
            random_seed=0,
            data_dir="./data",
            train_val_split=0.8,
            batch_size=5,
            tokenizer=prepare_batch,
            classifier=DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            ),
            optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        )
    )
