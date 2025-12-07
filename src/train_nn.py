import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset, random_split

cwd = os.getcwd()
if cwd.endswith("notebooks"):
    os.chdir("..")

DATA_PATH = Path("data/processed/bank_processed_unbalanced.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class marketing_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class simpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim=39):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def get_device():
    try:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            return torch.accelerator.current_accelerator().type
    except Exception:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_train_loader():
    df = pd.read_csv(DATA_PATH)

    minority = df[df["y_yes"] == 1]
    majority = df[df["y_yes"] == 0]

    majority_down = resample(
        majority,
        replace=False,
        n_samples=len(minority),
        random_state=42
    )

    df_bal = (
        pd.concat([majority_down, minority])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    X = df_bal.drop(columns=["y_yes"]).astype("float32").to_numpy()
    y = df_bal["y_yes"].to_numpy().astype("int64")

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = marketing_dataset(X_tensor, y_tensor)

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_set, _, _ = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    return train_loader


def train_model(train_loader, device, input_dim):
    model = simpleNeuralNetwork(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 45
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_loader.dataset):.4f}")

    return model


def main():
    device = get_device()
    train_loader = load_train_loader()

    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]

    model = train_model(train_loader, device, input_dim)

    out_path = MODEL_DIR / "neural_net.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
