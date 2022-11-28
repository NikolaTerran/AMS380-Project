import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_PATH = "data/ms_train"
TEST_PATH = "data/ms_test"

input_size = 1
hidden_size = 64
output_size = 1
num_epochs = 30
learning_rate = 0.001


class FinancialDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.data_frame = data
        self.data = data.drop(
            columns=[
                "Date",
                "High",
                "Low",
                "Stock Splits",
                "Volume",
                "Dividends",
            ]
        ).values
        self.close = data["Close"].values
        self.index = data["Close"].index
        self.close = (self.close - np.mean(self.close)) / np.sqrt(np.std(self.close))

    def __len__(self):
        return len(self.close) - input_size

    def __getitem__(self, index):
        return (
            self.close[index : index + input_size],
            self.close[index + input_size],
        )


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.LSTM(input_size, hidden_size, num_layers=3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.fc1(x)
        out = self.fc2(out)
        return out


def run_model(
    loader: DataLoader,
    model: nn.Module,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    train=False,
    num_epochs=1,
):
    if not train:
        num_epochs = 1
    for epoch in range(num_epochs):
        total_loss = 0
        indexes = []
        results = []
        outputs = []

        for i, (item, result) in enumerate(loader):
            output = model(item.to(torch.float32)).to(torch.float32)
            result = torch.unsqueeze(result.to(torch.float32), 0)

            if epoch == num_epochs - 1:
                indexes.append(i)
                results.append(result.item())
                outputs.append(output.item())

            loss = criterion(output, result)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss = torch.add(total_loss, loss.data)

        if epoch == num_epochs - 1:
            plt.scatter(indexes, outputs, s=1)
        print(epoch, total_loss)

    dataset = loader.dataset
    plt.scatter(dataset.index, dataset.close, label="Expected", s=1)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Construct our model, criterion, and optimizer
    model = FeedForwardNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create Datasets
    train_dataset = FinancialDataset(TRAIN_PATH)
    test_dataset = FinancialDataset(TEST_PATH)

    # Create Dataloaders
    train_loader = DataLoader(dataset=train_dataset)
    test_loader = DataLoader(dataset=test_dataset)

    # Train model
    run_model(train_loader, model, criterion, optimizer, train=True, num_epochs=num_epochs)

    # Run on validation data
    with torch.no_grad():
        run_model(test_loader, model, criterion, optimizer, train=False, num_epochs=num_epochs)
