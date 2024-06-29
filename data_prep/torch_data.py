import torch
from torch.utils.data import DataLoader, Dataset


class CTADataset(Dataset):
    def __init__(self, sequence, labels):
        self.sequence = sequence
        self.labels = labels

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.labels[idx]


def create_dataloader(train_data, validation_data, test_data, label_count):
    train_texts, train_labels = torch.tensor(
        train_data["victim_sequence"]
    ), torch.tensor(train_data["actor"])
    val_texts, val_labels = torch.tensor(
        validation_data["victim_sequence"]
    ), torch.tensor(validation_data["actor"])
    test_texts, test_labels = torch.tensor(test_data["victim_sequence"]), torch.tensor(
        test_data["actor"]
    )

    train_dataset = CTADataset(train_texts, train_labels)
    val_dataset = CTADataset(val_texts, val_labels)
    test_dataset = CTADataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    torch.save(train_loader, f"../data/train_loader_{label_count}.pt")
    torch.save(val_loader, f"../data/val_loader_{label_count}.pt")
    torch.save(test_loader, f"../data/test_loader_{label_count}.pt")
