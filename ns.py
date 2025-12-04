import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import time
import numpy as np



# === Генерация окон ===
def create_windows(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i : i+seq_len])
        Y.append(data[i+seq_len : i+seq_len+pred_len])  
    return np.array(X), np.array(Y)

# === Модель RNN ===
class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)              # [batch, seq_len, hidden_dim]
        out = out[:, -1, :]               # берём последний шаг
        out = self.fc(out)                # [batch, output_dim]
        return out


# === Загрузка и нормализация данных ===
def load_series(csv_path):
    df = pd.read_csv(csv_path)
    values = torch.tensor(df["value"].values, dtype=torch.float32)
    mean = values.mean().item()
    std = values.std().item()
    normed = np.array((values - mean) / std)
    return normed, mean, std


# === Функция обучения ===
def train(model, X, Y, optimizer = None, epochs=50, batch_size=32, lr=0.001, device = torch.device('cpu')):
    criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(device)
            yb = Y[i:i+batch_size].to(device)

            outputs = model(xb)
            loss = criterion(outputs.squeeze(), yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return optimizer

# === Функция предсказания ===
def predict(model, seq):
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    with torch.no_grad():
        pred = model(seq)
    return pred.squeeze(0)
