import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../data_prep"))

import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import load
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

warnings.filterwarnings("ignore")


class HybridArchitecture(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        max_len,
        num_layers,
        num_heads,
        hidden_size,
        dropout,
        num_filters,
        kernel_size,
    ):
        super(HybridArchitecture, self).__init__()
        self.token_embedding = nn.Embedding(input_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_len, hidden_size)
        self.conv1d = nn.Conv1d(
            hidden_size, num_filters, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.spatial_dropout = nn.Dropout2d(dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers
        )
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, sequence):
        sequence_length = sequence.size(1)
        position = torch.arange(sequence_length, device=sequence.device).unsqueeze(0)
        token_embedded = self.token_embedding(sequence)
        positional_embedded = self.positional_embedding(position)
        embedded = token_embedded + positional_embedded
        embedded = embedded.permute(0, 2, 1)
        x = self.conv1d(embedded)
        x = x + embedded
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)
        encoded = self.transformer_encoder(x)
        encoded = encoded + x
        pooled = encoded.mean(dim=1)
        logits = self.fc(pooled)
        return logits


def hyperparameters():
    dropouts = [0.1, 0.2]
    lrs = [1e-4, 3e-4, 1e-3, 1e-5, 2e-3]
    num_filter = 128
    num_heads = [4, 8, 16]
    num_layers = [2, 3, 4]
    epochs = [50, 100, 150, 200, 250]
    hidden_size = 128
    kernel_sizes = [3, 5, 7]

    grid_list = [
        {
            "dropout": dropout,
            "epoch": epoch,
            "hidden_size": hidden_size,
            "kernel_size": kernel_size,
            "lr": lr,
            "num_filters": num_filter,
            "num_heads": nh,
            "num_layers": nl,
        }
        for dropout in dropouts
        for epoch in epochs
        for kernel_size in kernel_sizes
        for lr in lrs
        for nh in num_heads
        for nl in num_layers
    ]

    return grid_list


def keep_actor_count(df, label_count):
    actor_names = df["actor"].value_counts().index.tolist()
    actor_names = actor_names[:label_count]
    df = df[df["actor"].isin(actor_names)]
    df = df.reset_index(drop=True)
    return df


def load_dataset(filepath, label_count):
    df = pd.read_parquet(filepath)
    actor_counts = df["actor"].value_counts()
    df = df[df["actor"] != "other"]  # remove other (wrongly labeled)
    df = df.reset_index(drop=True)
    df = keep_actor_count(df, label_count)
    return df


def tune_hyperparameters(label_count):
    df = load_dataset("../data/data_timestamped.parquet", label_count)

    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    relative_path = os.path.join(
        "experiment_results", f"hybridarchitecture_{now}_{label_count}.json"
    )
    experiment_filepath = os.path.join(os.getcwd(), relative_path)

    grid_list = hyperparameters()

    data_filepath = os.path.join(os.getcwd(), "../data")
    train_loader = torch.load(f"{data_filepath}/train_loader_{label_count}.pt")
    val_loader = torch.load(f"{data_filepath}/val_loader_{label_count}.pt")
    test_loader = torch.load(f"{data_filepath}/test_loader_{label_count}.pt")

    data_filepath = os.path.join(os.getcwd(), "../data/supportfiles")
    token2id = load(f"{data_filepath}/word2id_timestamped.joblib")
    id2token = load(f"{data_filepath}/id2word_timestamped.joblib")
    max_len = load(f"{data_filepath}/max_length_main_timestamped.joblib")

    num_classes = len(df["actor"].value_counts())
    print("Number of classes:", num_classes)

    unique_labels = []
    for idx, (sequences, labels) in enumerate(train_loader):
        unique_labels.extend(labels.tolist())
    unique_labels = set(unique_labels)

    print("Unique labels:", unique_labels)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    results = []
    for params in tqdm(grid_list, dynamic_ncols=True):
        model = HybridArchitecture(
            len(token2id) + 1,
            num_classes,
            max_len,
            num_layers=params["num_layers"],
            num_heads=params["num_heads"],
            hidden_size=params["hidden_size"],
            dropout=params["dropout"],
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()

        epochs = params["epoch"]

        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            model.train()
            for idx, (sequences, labels) in enumerate(train_loader):
                sequences = sequences.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(sequences)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                train_acc += (preds == labels).sum().item()
            train_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)
            model.eval()

            with torch.no_grad():
                y_val_true = []
                y_val_pred = []
                for batch in val_loader:
                    sequences, labels = batch
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    logits = model(sequences)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)
                    preds = torch.argmax(logits, dim=1)
                    val_acc += (preds == labels).sum().item()
                    y_val_true.extend(labels.tolist())
                    y_val_pred.extend(preds.tolist())

                val_loss /= len(val_loader.dataset)
                val_acc /= len(val_loader.dataset)
                val_f1 = f1_score(y_val_true, y_val_pred, average="weighted")

        test_loss = 0.0
        test_acc = 0.0
        y_test = []
        y_preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                sequences, labels = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                logits = model(sequences)
                loss = criterion(logits, labels)
                test_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                for i in range(len(labels)):
                    y_test.append(labels[i].item())
                    y_preds.append(preds[i].item())
                test_acc += (preds == labels).sum().item()
            test_loss /= len(test_loader.dataset)
            test_acc /= len(test_loader.dataset)
            test_f1 = f1_score(y_test, y_preds, average="weighted")

        print(
            "Best validation accuracy with parameters {} is {:.4f}, F1: {:.4f}, Test Accuracy: {:.4f}, Test F1: {:.4f}".format(
                params, val_acc, val_f1, test_acc, test_f1
            ),
            end="\r",
        )

        results.append(
            {
                "params": params,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }
        )

        results.sort(key=lambda x: x["val_f1"], reverse=True)

        with open(experiment_filepath, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    label_counts = [4, 10, 34]
    for label_count in label_counts:
        tune_hyperparameters(label_count)
