import os

import datasets
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def tokens2sentences(df):
    id2token = load("../data/supportfiles/id2word_timestamped.joblib")

    sentences = []
    for idx, row in df.iterrows():
        seq = row["victim_sequence"]
        sentences.append("")
        for s in seq:
            if s != 5429:
                sentences[-1] += id2token[s] + " "

    df = df.drop(columns=["victim_sequence"])
    df["sentence"] = sentences
    return df


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True)


def create_hf_dataset(
    tokenizer_name,
    file_name,
    train_data,
    validation_data,
    test_data,
    actor2label,
    label2actor,
):
    train_data = tokens2sentences(train_data)
    validation_data = tokens2sentences(validation_data)
    test_data = tokens2sentences(test_data)

    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"text": train_data["sentence"], "label": train_data["actor"]}
            ),
            "validation": datasets.Dataset.from_dict(
                {
                    "text": validation_data["sentence"],
                    "label": validation_data["actor"],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {"text": test_data["sentence"], "label": test_data["actor"]}
            ),
        }
    )

    feat_label = datasets.ClassLabel(
        names=list(actor2label.keys()), num_classes=len(actor2label)
    )
    dataset = dataset.cast_column("label", feat_label)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset_encoded = dataset.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        batch_size=None,
    )
    dataset_encoded.save_to_disk(f"../data/hf_{file_name}_timestamped_all")
