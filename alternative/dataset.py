from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class YawnDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences  # Shape: (num_sequences, seq_length, 64, 64)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        if self.transform:
            sequence = torch.stack([self.transform(frame) for frame in sequence])

        return sequence, label


# Example data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = YawnDataset(train_sequences, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
