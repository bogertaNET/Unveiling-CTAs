import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../experiments"))

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_metric
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from tune_pretrained import (compute_metrics, find_model_path, get_num_labels,
                             load_hf_dataset, load_model, load_tokenizer)


def load_best_hyperparameters(model_name, label_count):
    experiment_name = f"{model_name}_{label_count}"
    results_path = "../results.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    for result in results:
        if result["name"] == experiment_name:
            return (
                result["num_train_epochs"],
                result["batch_size"],
                result["learning_rate"],
                result["weight_decay"],
            )


def log_results(results_path, model_name, label_count, metrics):
    experiment_name = f"{model_name}_{label_count}"
    results_path = "../results.json"

    with open(results_path, "r") as f:
        logs = json.load(f)

    for log in logs:
        if log["name"] == experiment_name:
            log["cv_metrics"] = metrics
            break

    with open(results_path, "w") as f:
        json.dump(logs, f)


def cross_validation(model_name, label_count):
    dataset = load_hf_dataset(model_name, label_count)
    num_labels = get_num_labels(dataset)

    train_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    num_train_epochs, batch_size, learning_rate, weight_decay = (
        load_best_hyperparameters(model_name, label_count)
    )
    all_fold_acc = []
    all_fold_f1 = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(train_dataset, train_dataset["label"]):
        model = load_model(model_name, dataset, num_labels)
        tokenizer = load_tokenizer(model_name)
        train_dataset_fold = train_dataset.select(train_index)
        test_dataset_fold = train_dataset.select(test_index)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir="validation_results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_fold,
            eval_dataset=test_dataset_fold,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        eval_results = trainer.evaluate()

        all_fold_acc.append(eval_results["eval_accuracy"])
        all_fold_f1.append(eval_results["eval_f1"])

        del model
        del tokenizer
        del train_dataset
        del trainer
        del data_collator
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
    log_results("../results.json", model_name, label_count, metrics)


if __name__ == "__main__":
    model_names = ["BERT", "RoBERTa", "SecureBERT", "DarkBERT"]
    label_counts = [4, 10, 34]
    for model_name in model_names:
        for label_count in label_counts:
            cross_validation(model_name, label_count)
