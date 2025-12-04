import torch
import torch.nn as nn
import numpy as np

# === Генерация окон ===
def create_windows(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])  # одно число вперёд
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

# === Функция обучения ===
def train_model(model, X, Y, epochs=50, batch_size=32, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]

            outputs = model(xb)
            loss = criterion(outputs.squeeze(), yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# === Функция предсказания ===
def predict(model, seq):
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    with torch.no_grad():
        pred = model(seq)
    return pred.item()

# === Пример использования ===
if __name__ == "__main__":
    # Синтетический ряд (например, синус)
    data = np.sin(np.linspace(0, 20, 200))

    SEQ_LEN = 20
    X, Y = create_windows(data, SEQ_LEN)

    # В тензоры
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [samples, seq_len, 1]
    Y = torch.tensor(Y, dtype=torch.float32)                # [samples]

    model = RNNModel()

    # Обучение
    train_model(model, X, Y, epochs=100, batch_size=32, lr=0.01)

    # Предсказание
    test_seq = data[-SEQ_LEN:]  # последние 20 точек
    next_val = predict(model, test_seq)
    print("Следующее значение:", next_val)
