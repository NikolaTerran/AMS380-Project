import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Hyperparameters
input_size = 1 # Number of previus data used as input
hidden_size = 32 # Hidden layer size
output_size = 1 # Output size
num_epochs = 20 # Number of epochs
learning_rate = 0.001 # Learning rate (affects penalization)


class FinancialDataset(Dataset):
    """
    Dataset class that inherits from Pytorch's Dataset class. Takes input of
    path to CSV file containing data.
    """
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.data_frame = data
        self.close = data["Close"].values # Closing price as NumPy array
        self.index = data["Close"].index # Index of day
        self.mean = np.mean(self.close)
        self.stdev = np.std(self.close)

        self.close = (self.close - self.mean) / self.stdev # Normalize data

    def renormalize(self, mean: float, stdev: float):
        """
        Given data, renormalize it with given mean and standard deviation
        """
        self.close = (self.data_frame["Close"].values - mean) / stdev 

    def __len__(self):
        """
        Give the length of the dataset
        """
        return len(self.close) - input_size

    def __getitem__(self, index):
        """
        Given an index, return the corresponding element in the 
        tuple (input_value, expected_value)
        """
        # interval is [index, index + size), so it will not return index + input_size
        # for the input, and the expected result will be at index + input_size
        return (
            self.close[index : index + input_size], 
            self.close[index + input_size],
        )


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.LSTM(input_size, hidden_size, num_layers=3) # LSTM layer
        self.fc2 = nn.Linear(hidden_size, output_size) # Linear layer after

    def forward(self, x):
        out, (hn, cn) = self.fc1(x)
        out = self.fc2(out)
        return out


def run_model(
    loader: DataLoader,
    model: nn.Module,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    name: str,
    train=False,
    num_epochs=1,
    plot_start=0,
):
    if not train:
        # Only run once for tests
        num_epochs = 1
    for epoch in range(num_epochs):
        total_loss = 0
        indexes = []
        results = []
        outputs = []

        # Iterate through dataset and train the model
        for i, (item, result) in enumerate(loader):
            output = model(item.to(torch.float32)).to(torch.float32)
            # Need to make the dimensions the same as the input
            result = torch.unsqueeze(result.to(torch.float32), 0) 

            if epoch == num_epochs - 1:
                # Append results at the last epoch
                indexes.append(i)
                results.append(result.item())
                outputs.append(output.item())

            loss = criterion(output, result)
            if train:
                # Only run the optimizer/loss steps during training
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss = torch.add(total_loss, loss.data)

        if epoch == num_epochs - 1:
            # Plot the training data
            plt.scatter(indexes, outputs, s=1, label="Predicted")
        print(epoch, total_loss)

    dataset = loader.dataset
    plt.text(10, np.min(dataset.close + 1), "Loss = {0:.2f}".format(total_loss))
    plt.scatter(dataset.index, dataset.close, label="True value", s=1)
    plt.title("{0} {1}".format(name, "train" if train else "test"))
    plt.legend()
    plt.tight_layout()

    save_path = "plots/{0}_{1}.png".format(name, "train" if train else "test")
    plt.savefig(save_path)
    plt.close()


def run_stock_ticker(stock_name: str):
    model = FeedForwardNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Paths
    train_path = "data/{0}_train.csv".format(stock_name)
    test_path = "data/{0}_test.csv".format(stock_name)

    # Create Datasets
    train_dataset = FinancialDataset(train_path)
    test_dataset = FinancialDataset(test_path)
    test_dataset.renormalize(train_dataset.mean, train_dataset.stdev)
    
    # Create Dataloaders
    train_loader = DataLoader(dataset=train_dataset)
    test_loader = DataLoader(dataset=test_dataset)

    # Train model
    run_model(
        train_loader,
        model,
        criterion,
        optimizer,
        stock_name,
        train=True,
        num_epochs=num_epochs,
    )

    # Run on test data
    with torch.no_grad():
        run_model(
            test_loader,
            model,
            criterion,
            optimizer,
            stock_name,
            train=False,
            num_epochs=num_epochs,
        )


if __name__ == "__main__":
    stock_ticker_symbols = ["MSFT", "GOOG", "META", "MANU", "PSX"]
    for stock in stock_ticker_symbols:
        run_stock_ticker(stock)
