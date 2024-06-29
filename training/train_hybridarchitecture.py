import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../data_prep"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../experiments"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../validate"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import load
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from torch.utils.data import ConcatDataset, DataLoader
from tune_hybridarchitecture import HybridArchitecture, load_dataset
from validate_hybridarchitecture import load_best_hyperparameters


def log_results(results_path, label_count, metrics):
    experiment_name = f"Hybrid_{label_count}"
    results_path = "../results.json"

    with open(results_path, "r") as f:
        logs = json.load(f)

    for log in logs:
        if log["name"] == experiment_name:
            log["test_metrics"] = metrics
            break

    with open(results_path, "w") as f:
        json.dump(logs, f)


def plot_confusion_matrix(y_pred, y_test, unique_labels, title, label_count):
    unique, counts = np.unique(y_test, return_counts=True)
    label_count_dict = {lbl: count for lbl, count in zip(unique, counts)}

    sorted_labels = sorted(label_count_dict.keys(), key=lambda x: -label_count_dict[x])

    sorted_labels_str = [f"CTA {i+1}" for i in sorted_labels]

    cm = confusion_matrix(y_test, y_pred, labels=sorted_labels, normalize="true")

    fig, ax = plt.subplots(figsize=(16, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted_labels_str)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)

    plt.title(f"Model: HybridArchitecture | CTA Count: {label_count}")
    plt.xticks(rotation="vertical")
    plt.savefig(f"../plots/{title}.png")
    plt.show()


def train(label_count):
    train_loader = torch.load(f"../data/train_loader_{label_count}.pt")
    val_loader = torch.load(f"../data/val_loader_{label_count}.pt")
    test_loader = torch.load(f"../data/test_loader_{label_count}.pt")

    train_val_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    train_val_loader = DataLoader(train_val_dataset, batch_size=64, shuffle=True)

    token2id = load(f"../data/supportfiles/word2id_timestamped.joblib")
    id2token = load(f"../data/supportfiles/id2word_timestamped.joblib")
    max_len = load(f"../data/supportfiles/max_length_main_timestamped.joblib")

    print(f"Number of classes: {label_count}")

    unique_labels = []
    for idx, (sequences, labels) in enumerate(train_val_loader):
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

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for idx, (sequences, labels) in enumerate(train_val_loader):
            optimizer.zero_grad()
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
        train_loss /= len(train_val_loader.dataset)
        train_acc /= len(train_val_loader.dataset)
        print(
            f"Epoch {epoch + 1} - Train loss: {train_loss:.4f} - Train acc: {train_acc:.4f}"
        )

    test_loss = 0.0
    test_acc = 0.0
    y_pred = []
    y_test = []
    model.eval()
    with torch.no_grad():
        for idx, (sequences, labels) in enumerate(test_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).sum().item()
            y_pred.extend(outputs.argmax(1).tolist())
            y_test.extend(labels.tolist())
        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        print(
            f"Test loss: {test_loss:.4f} - Test acc: {test_acc:.4f} - Test f1: {test_f1:.4f}"
        )

    log_results("../results.json", label_count, {"accuracy": test_acc, "f1": test_f1})

    plot_confusion_matrix(
        y_pred, y_test, unique_labels, f"Hybrid_{label_count}", label_count
    )

    torch.save(model.state_dict(), f"../models/hybrid_{label_count}.pt")


if __name__ == "__main__":
    label_counts = [4, 10, 34]
    for label_count in label_counts:
        train(label_count)
