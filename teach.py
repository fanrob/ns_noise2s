import torch
import ns
import os

from constns import *

print("--------------------------------------------------------------")
print("")


# === Параметры ===

EPOCHS = 100    # количество эпох
LR = 5e-2    # скорость обучения
BETA = 0.001   # коэффициент для взвешивания потерь     


def main_RNN():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print("Using device:", device)


    #series, mean, std = ns.load_series(CSV_PATH)
    
    sX, sY, meanX, stdX, meanY, stdY = ns.load_series2(CSV_PATH)

    print(f"Loaded series of length {len(sX)}")
    
    #X, Y = ns.create_windows(series, SEQ_LEN, PRED_LEN)
    X, Y = ns.create_windows2(sX, sY, SEQ_LEN, PRED_LEN)

    # В тензоры
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [samples, seq_len, 1]
    Y = torch.tensor(Y, dtype=torch.float32)                # [samples]

    model = ns.RNNModel(input_dim=1, hidden_dim=MODEL_DIM, num_layers=NUM_LAYERS, output_dim=PRED_LEN).to(device)
    print("Model initialized.")

    #загрузить модель из файла если есть
    if "model.pth" in os.listdir():
        print("Loading model from file...")
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Model loaded.")
    else:
        print("No model file found, training new model.")

   
    #загрузить optimizer из файла если есть
    if "optimizer.pth" in os.listdir():
        print("Loading optimizer from file...")
        optimizer = torch.optim.Adam(model.parameters(), LR)
        checkpoint = torch.load("optimizer.pth", map_location=device)
        optimizer.load_state_dict(checkpoint)

        print("Optimizer loaded.")
        
        optimizer = ns.train(model, X, Y,optimizer=optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device = device)
    else:
        print("No optimizer file found, creating new optimizer.")
        optimizer = ns.train(model, X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device = device)
    
    # сохранить optimizer и модель в файл
    torch.save(optimizer.state_dict(), "optimizer.pth")
    torch.save(model.state_dict(), "model.pth")
  
    print("Training completed.")

def main_LSTM():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


    sX, sY, meanX, stdX, meanY, stdY = ns.load_series2(CSV_PATH)
    print(f"Loaded series of length {len(sX)}")
    X, Y = ns.create_windows2(sX, sY, SEQ_LEN, PRED_LEN)
    # В тензоры
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Add channel dimension -> [samples, seq_len, 1] and [samples, pred_len, 1]
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)

    # Prepare windows as list of (input_seq, target_seq) tuples
    windows = [(X[i], Y[i]) for i in range(len(X))]


    model = ns.LSTMModel(input_dim=1, hidden_dim=MODEL_DIM, num_layers=NUM_LAYERS, pred_len=PRED_LEN).to(device)
    print("Model initialized.")
    #загрузить модель из файла если есть
    if "model_lstm.pth" in os.listdir():
        print("Loading model from file...")
        model.load_state_dict(torch.load("model_lstm.pth", map_location=device))
        print("Model loaded.")
    else:
        print("No model file found, training new model.")
    #загрузить optimizer из файла если есть
    if "optimizer_lstm.pth" in os.listdir():
        print("Loading optimizer from file...")
        optimizer = torch.optim.Adam(model.parameters(), LR)
        checkpoint = torch.load("optimizer_lstm.pth", map_location=device)
        optimizer.load_state_dict(checkpoint)
        print("Optimizer loaded.")
        optimizer = ns.trainLSTM(model, windows,optimizer=optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device = device)
    else:
        print("No optimizer file found, creating new optimizer.")
        optimizer = ns.trainLSTM(model, windows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device = device)
    # сохранить optimizer и модель в файл
    torch.save(optimizer.state_dict(), "optimizer_lstm.pth")
    torch.save(model.state_dict(), "model_lstm.pth")
    print("Training completed.")
    

# === Главный блок ===
if __name__ == "__main__":
    #main_RNN()
    
    main_LSTM()


   
    

    # Предсказание
    # test_seq = series[-SEQ_LEN:]  # последние 20 точек
    # next_val = ns.predict(model, test_seq)
    # print("Следующее значение:", next_val)

    
