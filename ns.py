import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import time
import numpy as np


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



# === Модель LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=4, num_layers=4, pred_len=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len)  # выдаём сразу PRED_LEN значений

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)                       # [batch, seq_len, hidden]
        out = self.fc(out[:, -1, :])                # берём последний шаг → [batch, pred_len]
        return out.unsqueeze(-1)                    # [batch, pred_len, 1]

# === Функция обучения ===
def trainLSTM(model, windows, optimizer = None, epochs=20, lr=5e-4, batch_size=32, device = torch.device('cpu')):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    model.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        batch_losses = []

        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            inputs = torch.stack([w[0] for w in batch]).to(device)   # [batch, seq_len, 1]
            targets = torch.stack([w[1] for w in batch]).to(device)  # [batch, pred_len, 1]

            outputs = model(inputs)                                  # [batch, pred_len, 1]
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        print(f"Epoch {epoch+1}, Loss: {sum(batch_losses)/len(batch_losses):.4f}, "
              f"Time: {time.time()-start_time:.2f}s")

    return optimizer

# === Функция предсказания ===
def predictLSTM(model, seq, device = torch.device('cpu')):
    model.eval()
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
    with torch.no_grad():
        pred = model(seq)  # [1, pred_len, 1]
    return pred.squeeze().cpu().numpy()  # [pred_len]






# === Генерация окон ===
def create_windows(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i : i+seq_len])
        Y.append(data[i+seq_len : i+seq_len+pred_len])  
    return np.array(X), np.array(Y)


# === Генерация окон ===
def create_windows2(dataX, dataY, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(dataX) - seq_len):
        X.append(dataX[i : i+seq_len])
        Y.append(dataY[i : i+pred_len])  
    return np.array(X), np.array(Y)



# === Загрузка и нормализация данных ===
def load_series(csv_path):
    df = pd.read_csv(csv_path)
    values = torch.tensor(df["value"].values, dtype=torch.float32)
    mean = values.mean().item()
    std = values.std().item()
    normed = np.array((values - mean) / std)
    return normed, mean, std

# === Загрузка и нормализация данных ===
def load_series2(csv_path):
    df = pd.read_csv(csv_path)
    valuesX = torch.tensor(df["time"].values, dtype=torch.float32)
    valuesY = torch.tensor(df["value"].values, dtype=torch.float32)
    meanX = valuesX.mean().item()
    stdX = valuesX.std().item()
    meanY = valuesY.mean().item()
    stdY = valuesY.std().item()
    X = np.array((valuesX - meanX) / stdX)
    Y = np.array((valuesY - meanY) / stdY)
    return X, Y, meanX, stdX, meanY, stdY



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
