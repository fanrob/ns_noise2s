import torch
import matplotlib.pyplot as plt
import ns
import numpy as np
from constns import *

print("--------------------------------------------------------------")
print("")


# === Параметры ===

START_POINT = 0  #начальная позиция входных данных
WINDOW = 1000 # какой кусок данных берем для рисунка графика и предсказания


# === Главный блок ===
if __name__ == "__main__":
    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Загрузка и подготовка данных
    series, mean, std = ns.load_series(CSV_PATH)

    
    print(f"Loaded series of length {len(series)}") 
    
    model = ns.RNNModel(input_dim=1, hidden_dim=MODEL_DIM, num_layers=NUM_LAYERS, output_dim=PRED_LEN)
    model.load_state_dict(torch.load("model.pth"))
    print("Model initialized.")
  
    # Прогноз на основе последних SEQ_LEN значений
    graph1 = series[START_POINT:START_POINT+WINDOW]  # исторические данные для графика
    #копия для предсказаний
    graph2 = graph1.copy() # предсказанные данные для графика
       
    for i in range(0,WINDOW-SEQ_LEN-PRED_LEN,PRED_LEN):
        known = graph1[i:i+SEQ_LEN]       
        predict = ns.predict(model, known)     # предсказываем !!!
        graph2[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN] = predict  # сохраняем в график предсказаний

    graph1 = graph1 * std + mean  # денормализация
    graph2 = graph2 * std + mean  # денормализация
    

   

    # Включаем тёмный стиль
    plt.style.use("dark_background")

    # Визуализация
    plt.figure(figsize=(10, 5))

    # Исторические данные
    plt.plot(range(len(graph1)), graph1, label="Исторические данные", color="cyan", linewidth=1)

    # Предсказания
    plt.plot(range(len(graph2)), graph2, label="Предсказания модели", color="orange", linewidth=1)


    # Настройки
    plt.legend(facecolor="black", edgecolor="white")
    plt.title("Currency Forecast", color="white")
    plt.xlabel("Time step", color="white")
    plt.ylabel("Value", color="white")
    step = 10
    plt.xticks(np.arange(0, len(graph1), step))
    plt.grid(True, color="gray", alpha=0.3)
    plt.tight_layout()
    plt.show()
        
