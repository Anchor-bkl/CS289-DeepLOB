#!/usr/bin/env python3
"""
Exact FI-2010 reproduction of the original PyTorch DeepLOB notebook.

This script intentionally mirrors the training setup used in:
https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/
blob/master/jupyter_pytorch/run_train_pytorch.ipynb

Reproduced notebook choices:
- FI-2010 decimal-precision dataset
- 80/20 split on Train_Dst_NoAuction_DecPre_CF_7.txt
- test set = Test_Dst_NoAuction_DecPre_CF_7/8/9 concatenated
- single horizon k=4 (the 100-event label, 0-based indexing)
- lookback T=100
- batch size 64
- Adam(lr=1e-4)
- CrossEntropyLoss
- 50 epochs with best-validation-loss checkpointing
- model forward returns softmax probabilities before the CE loss, matching the notebook exactly
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.environ.get(
    "FI_ORIGINAL_MODEL_DIR",
    os.path.join(BASE_DIR, "models", "fi_original_notebook"),
)
RESULT_DIR = os.environ.get(
    "FI_ORIGINAL_RESULT_DIR",
    os.path.join(BASE_DIR, "results", "fi_original_notebook"),
)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

CONFIG = {
    "k_idx": 4,
    "events_ahead": 100,
    "lookback": 100,
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-4,
    "criterion": "CrossEntropyLoss",
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "dropout": 0.0,
    "forward_output": "softmax_probabilities",
    "source_notebook": "zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books:jupyter_pytorch/run_train_pytorch.ipynb",
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_x(raw_data):
    return np.array(raw_data[:40, :].T)


def get_label(raw_data):
    lob = raw_data[-5:, :].T
    return lob


def data_classification(x_data, y_data, lookback):
    n_rows, n_features = x_data.shape
    data_y = np.array(y_data)[lookback - 1:n_rows]
    data_x = np.zeros((n_rows - lookback + 1, lookback, n_features))
    for idx in range(lookback, n_rows + 1):
        data_x[idx - lookback] = x_data[idx - lookback:idx, :]
    return data_x, data_y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch."""

    def __init__(self, raw_data, k, num_classes, lookback):
        self.k = k
        self.num_classes = num_classes
        self.lookback = lookback

        x_data = prepare_x(raw_data)
        y_data = get_label(raw_data)
        x_data, y_data = data_classification(x_data, y_data, self.lookback)
        y_data = y_data[:, self.k] - 1
        self.length = len(x_data)

        x_tensor = torch.from_numpy(x_data)
        self.x = torch.unsqueeze(x_tensor, 1)
        self.y = torch.from_numpy(y_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x_data):
        h0 = torch.zeros(1, x_data.size(0), 64).to(device)
        c0 = torch.zeros(1, x_data.size(0), 64).to(device)

        x_data = self.conv1(x_data)
        x_data = self.conv2(x_data)
        x_data = self.conv3(x_data)

        x_inp1 = self.inp1(x_data)
        x_inp2 = self.inp2(x_data)
        x_inp3 = self.inp3(x_data)
        x_data = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x_data = x_data.permute(0, 2, 1, 3)
        x_data = torch.reshape(x_data, (-1, x_data.shape[1], x_data.shape[2]))
        x_data, _ = self.lstm(x_data, (h0, c0))
        x_data = x_data[:, -1, :]
        x_data = self.fc1(x_data)
        forecast_y = torch.softmax(x_data, dim=1)
        return forecast_y


def load_data():
    dec_data = np.loadtxt(os.path.join(DATA_DIR, "Train_Dst_NoAuction_DecPre_CF_7.txt"))
    split_col = int(np.floor(dec_data.shape[1] * 0.8))
    dec_train = dec_data[:, :split_col]
    dec_val = dec_data[:, split_col:]

    dec_test1 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_7.txt"))
    dec_test2 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_8.txt"))
    dec_test3 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_9.txt"))
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    return dec_train, dec_val, dec_test


def batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs, model_path):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    best_val_loss = np.inf
    best_val_epoch = 0

    for epoch_idx in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.int64)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        train_losses[epoch_idx] = train_loss
        val_losses[epoch_idx] = val_loss

        if val_loss < best_val_loss:
            torch.save(model, model_path)
            best_val_loss = val_loss
            best_val_epoch = epoch_idx
            print("model saved")

        dt = datetime.now() - t0
        print(
            f"Epoch {epoch_idx + 1}/{epochs}, Train Loss: {train_loss:.4f},           "
            f"Validation Loss: {val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_val_epoch}"
        )

    return train_losses, val_losses, best_val_epoch + 1, float(best_val_loss)


def evaluate(model_path, test_loader):
    model = torch.load(model_path, map_location=device)
    model.eval()

    n_correct = 0.0
    n_total = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    test_acc = n_correct / n_total
    report = classification_report(all_targets, all_predictions, digits=4)
    report_dict = classification_report(all_targets, all_predictions, digits=4, output_dict=True)
    return test_acc, all_targets, all_predictions, report, report_dict


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print(device)
    dec_train, dec_val, dec_test = load_data()
    print(dec_train.shape, dec_val.shape, dec_test.shape)

    batch_size = CONFIG["batch_size"]
    dataset_train = Dataset(raw_data=dec_train, k=CONFIG["k_idx"], num_classes=3, lookback=CONFIG["lookback"])
    dataset_val = Dataset(raw_data=dec_val, k=CONFIG["k_idx"], num_classes=3, lookback=CONFIG["lookback"])
    dataset_test = Dataset(raw_data=dec_test, k=CONFIG["k_idx"], num_classes=3, lookback=CONFIG["lookback"])

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    print(dataset_train.x.shape, dataset_train.y.shape)

    model = deeplob(y_len=dataset_train.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    model_path = os.path.join(MODEL_DIR, "best_val_model_pytorch.pt")
    train_losses, val_losses, best_val_epoch, best_val_loss = batch_gd(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        epochs=CONFIG["epochs"],
        model_path=model_path,
    )

    loss_plot_path = os.path.join(RESULT_DIR, "original_notebook_loss.png")
    plot_losses(train_losses, val_losses, loss_plot_path)

    test_acc, all_targets, all_predictions, report_text, report_dict = evaluate(model_path, test_loader)
    print(f"Test acc: {test_acc:.4f}")
    print("accuracy_score:", accuracy_score(all_targets, all_predictions))
    print(report_text)

    np.savez_compressed(
        os.path.join(RESULT_DIR, "original_notebook_losses.npz"),
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_epoch=best_val_epoch,
        best_val_loss=best_val_loss,
    )
    np.savez_compressed(
        os.path.join(RESULT_DIR, "original_notebook_preds.npz"),
        y_true=all_targets,
        y_pred=all_predictions,
    )

    with open(os.path.join(RESULT_DIR, "original_notebook_report.txt"), "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    metrics = {
        "config": CONFIG,
        "device": str(device),
        "train_shape": list(dec_train.shape),
        "val_shape": list(dec_val.shape),
        "test_shape": list(dec_test.shape),
        "dataset_train_shape": list(dataset_train.x.shape),
        "dataset_train_target_shape": list(dataset_train.y.shape),
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val_loss,
        "test_acc": float(test_acc),
        "accuracy_score": float(accuracy_score(all_targets, all_predictions)),
        "classification_report": report_dict,
        "model_path": model_path,
        "loss_plot_path": loss_plot_path,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(os.path.join(RESULT_DIR, "original_notebook_metrics.json"), "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Saved original-notebook artifacts to {RESULT_DIR}")


if __name__ == "__main__":
    main()