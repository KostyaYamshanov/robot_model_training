import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
import os

# Предполагаем, что данные уже загружены, но для полноты начнем с загрузки
# Загрузка данных из всех CSV-файлов в папках
train_dir = 'data/train'
val_dir = 'data/validation'
test_dir = 'data/test'

def load_data_from_directory(directory):
    # Находим все файлы output.csv в подпапках
    files = glob.glob(os.path.join(directory, '*/output.csv'))
    # Читаем каждый файл и объединяем в один DataFrame
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return data

# Загружаем данные
train_data = load_data_from_directory(train_dir)
val_data = load_data_from_directory(val_dir)
test_data = load_data_from_directory(test_dir)

# Проверяем, что данные загрузились (первые 5 строк для примера)
print("Первые 5 строк обучающих данных:")
print(train_data.head())

# 1. Подготовка данных
# Мы хотим предсказывать следующую скорость (V_x, V_y, V_z) на основе текущего состояния
def prepare_data(df):
    # Выбираем входные признаки (текущее состояние)
    X_columns = ['V_x', 'V_y', 'V_z', 'rollspeed', 'pitchspeed', 'yawspeed', 
                 'servo1_raw', 'servo2_raw', 'servo3_raw', 'servo4_raw']
    # Целевые данные — скорости на следующем шаге
    y_columns = ['V_x', 'V_y', 'V_z']
    
    # Берем все строки кроме последней для X (т.к. для последней нет "следующего шага")
    X = df[X_columns].iloc[:-1].values
    # Берем все строки начиная со второй для y (сдвиг на 1 вперед)
    y = df[y_columns].iloc[1:].values
    
    return X, y

# Применяем функцию к каждому набору данных
X_train, y_train = prepare_data(train_data)
X_val, y_val = prepare_data(val_data)
X_test, y_test = prepare_data(test_data)

# Проверяем размеры
print("Размеры обучающих данных:", X_train.shape, y_train.shape)
print("Размеры валидационных данных:", X_val.shape, y_val.shape)
print("Размеры тестовых данных:", X_test.shape, y_test.shape)

# 2. Нормализация данных
# Нормализация нужна, чтобы привести все числа к одному масштабу
scaler_X = StandardScaler()
# Учим scaler на обучающих данных и сразу нормализуем их
X_train_scaled = scaler_X.fit_transform(X_train)
# Применяем тот же scaler к валидационным и тестовым данным
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
# Учим scaler на обучающих целях и нормализуем их
y_train_scaled = scaler_y.fit_transform(y_train)
# Применяем к валидационным и тестовым целям
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# 3. Преобразование данных в тензоры PyTorch
# PyTorch работает с тензорами, поэтому преобразуем numpy-массивы
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# 4. Создание DataLoader
# DataLoader разбивает данные на батчи для эффективного обучения
batch_size = 32

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Проверяем количество батчей
print("Количество батчей в train_loader:", len(train_loader))
print("Количество батчей в val_loader:", len(val_loader))
print("Количество батчей в test_loader:", len(test_loader))

# 5. Определение нейросети
class SpeedPredictor(nn.Module):
    def __init__(self):
        super(SpeedPredictor, self).__init__()
        # Входной слой: 10 признаков -> 64 нейрона
        self.fc1 = nn.Linear(10, 64)
        # Скрытый слой: 64 -> 32
        self.fc2 = nn.Linear(64, 32)
        # Выходной слой: 32 -> 3 (V_x, V_y, V_z)
        self.fc3 = nn.Linear(32, 3)
        # Функция активации
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # Первый слой с активацией
        x = self.relu(self.fc2(x))  # Второй слой с активацией
        x = self.fc3(x)             # Выходной слой (без активации для регрессии)
        return x

# Создаем модель
model = SpeedPredictor()

# 6. Настройка обучения
# Функция потерь — среднеквадратичная ошибка
criterion = nn.MSELoss()
# Оптимизатор — Adam с шагом обучения 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Обучение модели с ранней остановкой
num_epochs = 100
best_val_loss = float('inf')  # Лучшая ошибка на валидации
patience = 10  # Сколько эпох ждать без улучшения
patience_counter = 0

for epoch in range(num_epochs):
    # Обучение
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Обнуляем градиенты
        output = model(X_batch)  # Предсказание
        loss = criterion(output, y_batch)  # Ошибка
        loss.backward()  # Вычисляем градиенты
        optimizer.step()  # Обновляем веса
        train_loss += loss.item()
    
    train_loss /= len(train_loader)  # Средняя ошибка на батч
    
    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Без градиентов для экономии памяти
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Печатаем прогресс
    print(f'Эпоха {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Ранняя остановка
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')  # Сохраняем лучшую модель
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Ранняя остановка: обучение остановлено")
            break

# Загружаем лучшую модель
model.load_state_dict(torch.load('best_model.pth'))

# 8. Тестирование модели
model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        test_loss += loss.item()
    
test_loss /= len(test_loader)
print(f'Тестовая ошибка (MSE): {test_loss:.6f}')

# Пример предсказания для первой тестовой точки
with torch.no_grad():
    first_test_input = X_test_tensor[0].unsqueeze(0)  # Добавляем размер батча
    pred_scaled = model(first_test_input)
    pred = scaler_y.inverse_transform(pred_scaled.numpy())  # Обратная нормализация
    real = scaler_y.inverse_transform(y_test_tensor[0].unsqueeze(0).numpy())
    print("Пример предсказания (первая тестовая точка):")
    print("Предсказанные скорости:", pred[0])
    print("Реальные скорости:", real[0])
