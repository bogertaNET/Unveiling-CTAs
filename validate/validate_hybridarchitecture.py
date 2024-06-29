import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../data_prep"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../experiments"))

import numpy as np
import torch
from joblib import load
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tune_hybridarchitecture import HybridArchitecture, load_dataset


def load_best_hyperparameters(label_count):
    experiment_name = f"Hybrid_{label_count}"
    results_path = "../results.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    for result in results:
        if result["name"] == experiment_name:
            dropout = result["dropout"]
            epochs = result["epochs"]
            hidden_size = result["hidden_size"]
            kernel_size = result["kernel_size"]
            lr = result["lr"]
            num_filters = result["num_filters"]
            num_heads = result["num_heads"]
            num_layers = result["num_layers"]
            return (
                dropout,
                epochs,
                hidden_size,
                kernel_size,
                lr,
                num_filters,
                num_heads,
                num_layers,
            )


def log_results(results_path, label_count, metrics):
    experiment_name = f"Hybrid_{label_count}"
    results_path = "../results.json"

    with open(results_path, "r") as f:
        logs = json.load(f)

    for log in logs:
        if log["name"] == experiment_name:
            log["cv_metrics"] = metrics
            break

    with open(results_path, "w") as f:
        json.dump(logs, f)


def cross_validation(label_count):
    train_loader = torch.load(f"../data/train_loader_{label_count}.pt")
    val_loader = torch.load(f"../data/val_loader_{label_count}.pt")
    test_loader = torch.load(f"../data/test_loader_{label_count}.pt")

    labels_train = [label for _, label in train_loader.dataset]
    labels_val = [label for _, label in val_loader.dataset]
    labels_holder = labels_train + labels_val

    train_val_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])

    token2id = load("../data/supportfiles/word2id_timestamped.joblib")
    id2token = load("../data/supportfiles/id2word_timestamped.joblib")
    max_len = load("../data/supportfiles/max_length_main_timestamped.joblib")

    print(f"Number of classes: {label_count}")

    unique_labels = []
    for idx, (sequences, labels) in enumerate(train_loader):
        unique_labels.extend(labels.tolist())
    unique_labels = list(set(unique_labels))

    print(f"Unique labels: {unique_labels}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    (
        dropout,
        epochs,
        hidden_size,
        kernel_size,
        lr,
        num_filters,
        num_heads,
        num_layers,
    ) = load_best_hyperparameters(label_count)
    batch_size = 64
    all_fold_acc = []
    all_fold_f1 = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(train_val_dataset, labels_holder)
    ):
        print(f"Fold {fold + 1}")
        model = HybridArchitecture(
            len(token2id) + 1,
            label_count,
            max_len,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout,
            num_filters=num_filters,
            kernel_size=kernel_size,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_dataset = Subset(train_val_dataset, train_idx)
        val_dataset = Subset(train_val_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (outputs.argmax(1) == labels).sum().item()
            train_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)

            model.eval()
            with torch.no_grad():
                y_val_true = []
                y_val_pred = []
                for batch in val_loader:
                    sequences = batch[0].to(device)
                    labels = batch[1].to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.argmax(1) == labels).sum().item()
                    y_val_true.extend(labels.tolist())
                    y_val_pred.extend(outputs.argmax(1).tolist())
                val_loss /= len(val_loader.dataset)
                val_acc /= len(val_loader.dataset)
                val_f1 = f1_score(y_val_true, y_val_pred, average="weighted")

        all_fold_acc.append(val_acc)
        all_fold_f1.append(val_f1)

        del model
        del optimizer
        del criterion
    avg_acc = np.mean(all_fold_acc)
    std_acc = np.std(all_fold_acc)
    avg_f1 = np.mean(all_fold_f1)
    std_f1 = np.std(all_fold_f1)

    metrics = {
        "avg_val_acc": avg_acc,
        "std_val_acc": std_acc,
        "avg_val_f1": avg_f1,
        "std_val_f1": std_f1,
    }
    log_results("../results.json", label_count, metrics)


if __name__ == "__main__":
    label_counts = [4, 10, 34]
    for label_count in label_counts:
        cross_validation(label_count)
