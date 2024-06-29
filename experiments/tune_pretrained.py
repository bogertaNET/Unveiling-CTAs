import json
import os
import time

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


class ExperimentLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.experiments = []

    def log_experiment(self, hyperparameters, val_metrics, test_metrics):
        experiment = {
            "hyperparameters": hyperparameters,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        self.experiments.append(experiment)

    def save_experiments(self):
        self.experiments.sort(
            key=lambda x: x["validation_metrics"]["test_f1"], reverse=True
        )
        with open(self.log_file, "w") as f:
            json.dump(self.experiments, f, indent=4)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def get_experiment_path(model_name, label_count):
    now = str(time.time())
    abs_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_path)
    experiment_path = os.path.join(dir_name, "experiment_results")
    os.makedirs(experiment_path, exist_ok=True)
    experiment_path = os.path.join(
        experiment_path, f"{model_name}_{now}_{label_count}.json"
    )
    return experiment_path


def load_hf_dataset(model_name, label_count):
    abs_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_path)
    dir_name = os.path.dirname(dir_name)
    hf_datasetpath = os.path.join(
        dir_name, "data", f"hf_{model_name}_{label_count}_timestamped_all"
    )
    dataset = datasets.load_from_disk(hf_datasetpath)
    return dataset


def get_num_labels(dataset):
    return len(dataset["train"].features["label"].names)


def find_model_path(model_name):
    if model_name == "DarkBERT":
        model_path = os.path.expanduser("~/huggingface_models/DarkBERT")
    elif model_name == "RoBERTa":
        model_path = "roberta-base"
    elif model_name == "BERT":
        model_path = "bert-base-uncased"
    elif model_name == "SecureBERT":
        model_path = "ehsanaghaei/SecureBERT"
    return model_path


def load_model(model_name, dataset, num_labels):
    model_path = find_model_path(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    return model


def load_tokenizer(model_name):
    model_path = find_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def hyperparameters():
    epochs = [4, 6]
    batch_sizes = [16, 32]
    learning_rates = [1e-5, 2e-5, 3e-5]
    weight_decays = [0.01, 0.05]
    hyperparameters = []
    for epoch in epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for weight_decay in weight_decays:
                    hyperparameters.append(
                        {
                            "num_train_epochs": epoch,
                            "per_device_train_batch_size": batch_size,
                            "per_device_eval_batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                        }
                    )

    return hyperparameters


def tune(model_name, label_count):
    dataset = load_hf_dataset(model_name, label_count)
    num_labels = get_num_labels(dataset)
    logger = ExperimentLogger(get_experiment_path(model_name, num_labels))
    hyps = hyperparameters()

    for hyp in tqdm(hyps):
        model = load_model(model_name, dataset, num_labels)
        tokenizer = load_tokenizer(model_name)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyp["num_train_epochs"],
            per_device_train_batch_size=hyp["per_device_train_batch_size"],
            per_device_eval_batch_size=hyp["per_device_eval_batch_size"],
            learning_rate=hyp["learning_rate"],
            weight_decay=hyp["weight_decay"],
            evaluation_strategy="epoch",
            disable_tqdm=True,
            push_to_hub=False,
            log_level="error",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer.train()

        val_output = trainer.predict(dataset["validation"])
        val_metrics = val_output.metrics

        test_output = trainer.predict(dataset["test"])
        test_metrics = test_output.metrics

        logger.log_experiment(hyp, val_metrics, test_metrics)
        logger.save_experiments()

        del model
        del tokenizer
        del trainer
        del val_output
        del test_output


if __name__ == "__main__":
    model_names = ["BERT", "RoBERTa", "SecureBERT", "DarkBERT"]
    label_counts = [4, 10, 34]
    for model_name in model_names:
        for label_count in label_counts:
            tune(model_name, label_count)
