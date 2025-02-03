import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from hf_dataset import create_hf_dataset
from joblib import dump, load
from sklearn.model_selection import train_test_split
from torch_data import create_dataloader


def load_dataset(filepath):
    df = pd.read_parquet(filepath)
    actor_counts = df["actor"].value_counts()
    df = df[df["actor"] != "other"]  # remove other (wrongly labeled)
    df = df.reset_index(drop=True)
    print(df["actor"].value_counts())
    return df


def load_highest_n_sequence_actors(df, n):
    actor_counts = df["actor"].value_counts()
    actor_counts = actor_counts[actor_counts > n]
    df = df[df["actor"].isin(actor_counts.index)]
    print(f"Loaded {len(df['actor'].unique())} actors with more than {n} sequences")
    print(df["actor"].value_counts())
    return df


def give_actors_labels(df):
    actors = list(df["actor"].unique())
    actor2label = {actor: idx for idx, actor in enumerate(actors)}
    label2actor = {idx: actor for idx, actor in enumerate(actors)}

    df["actor"] = df["actor"].map(actor2label)
    return df, actor2label, label2actor


def train_val_test_split(df):
    train_data, temp_data = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["actor"], shuffle=True
    )
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        stratify=temp_data["actor"],
        shuffle=True,
    )

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)

    print(f"Train Data Shape: {train_data.shape}")
    print(f"Validation Data Shape: {validation_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    return train_data, validation_data, test_data


def create_datasets(df, label_count):
    df, actor2label, label2actor = give_actors_labels(df)
    train_data, validation_data, test_data = train_val_test_split(df)

    create_hf_dataset(
        "roberta-base",
        f"RoBERTa_{label_count}",
        train_data,
        validation_data,
        test_data,
        actor2label,
        label2actor,
    )
    create_hf_dataset(
        "bert-base-uncased",
        f"BERT_{label_count}",
        train_data,
        validation_data,
        test_data,
        actor2label,
        label2actor,
    )
    create_hf_dataset(
        "ehsanaghaei/SecureBERT",
        f"SecureBERT_{label_count}",
        train_data,
        validation_data,
        test_data,
        actor2label,
        label2actor,
    )
    create_hf_dataset(
        os.path.expanduser("~/huggingface_models/DarkBERT"),
        f"DarkBERT_{label_count}",
        train_data,
        validation_data,
        test_data,
        actor2label,
        label2actor,
    )

    create_dataloader(train_data, validation_data, test_data, label_count)

    dump(actor2label, "../data/supportfiles/actor2label.joblib")
    dump(label2actor, "../data/supportfiles/label2actor.joblib")


if __name__ == "__main__":
    df = load_dataset("../data/data_timestamped.parquet")
    df_1000 = load_highest_n_sequence_actors(df, 1000)
    df_50 = load_highest_n_sequence_actors(df, 50)
    df_10 = load_highest_n_sequence_actors(df, 10)

    create_datasets(df_1000, len(df_1000["actor"].unique()))
    create_datasets(df_50, len(df_50["actor"].unique()))
    create_datasets(df_10, len(df_10["actor"].unique()))
