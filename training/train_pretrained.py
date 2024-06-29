import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../experiments"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../validate"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from tune_pretrained import (compute_metrics, find_model_path, get_num_labels,
                             load_hf_dataset, load_model, load_tokenizer)
from validate_pretrained import load_best_hyperparameters


def log_test_results(results_path, model_name, label_count, metrics):
    experiment_name = f"{model_name}_{label_count}"
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


def train(model_name, label_count):
    dataset = load_hf_dataset(model_name, label_count)
    num_labels = get_num_labels(dataset)
    model = load_model(model_name, dataset, num_labels)
    tokenizer = load_tokenizer(model_name)

    train_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    num_train_epochs, batch_size, learning_rate, weight_decay = (
        load_best_hyperparameters(model_name, label_count)
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=f"../models/{model_name}_{label_count}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        disable_tqdm=True,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds_output = trainer.predict(dataset["test"])
    log_test_results("../results.json", model_name, label_count, preds_output.metrics)

    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_test = np.array(dataset["test"]["label"])
    labels = dataset["test"].features["label"].names
    plot_confusion_matrix(y_pred, y_test, labels, model_name, label_count)

    trainer.save_model(f"../models/{model_name}_{label_count}")

    del model
    del tokenizer
    del trainer


if __name__ == "__main__":
    model_names = ["BERT", "RoBERTa", "SecureBERT", "DarkBERT"]
    label_counts = [4, 10, 34]
    for model_name in model_names:
        for label_count in label_counts:
            train(model_name, label_count)
