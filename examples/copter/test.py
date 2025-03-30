import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------
# 1. Загрузка данных
# --------------------------------------
def load_data(file_path):
    """Загрузка данных из файла"""
    import ast
    with open(file_path, "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
    return data

# --------------------------------------
# 2. Восстановление траектории
# --------------------------------------
def reconstruct_trajectory(data):
    """Основная функция восстановления траектории"""
    # Проверка наличия обязательных ключей
    assert all(k in data for k in ['vx', 'vy', 'vz']), "Отсутствуют данные о скоростях!"
    
    # Временные метки
    if 'timestamp' in data:
        t = np.array(data['timestamp'])
        dt = np.gradient(t)
    else:
        t = np.arange(len(data['vx']))
        dt = np.ones_like(t)

    # Начальные условия
    initial_pos = {
        'x': data.get('x_initial', 0),
        'y': data.get('y_initial', 0), 
        'z': data.get('z_initial', 0)
    }

    # Преобразование скоростей в глобальную СК (если есть данные об ориентации)
    if all(k in data for k in ['roll', 'pitch', 'yaw']):
        # Создание матриц поворота
        rot = Rotation.from_euler(
            'zyx', 
            np.column_stack([data['yaw'], data['pitch'], data['roll']]),
            degrees=False
        )
        
        # Преобразование скоростей
        v_global = rot.apply(np.column_stack([data['vx'], data['vy'], data['vz']]))
    else:
        # Используем скорости как глобальные
        v_global = np.column_stack([data['vx'], data['vy'], data['vz']])

    # Интегрирование скоростей
    x = initial_pos['x'] + np.cumsum(v_global[:,0] * dt)
    y = initial_pos['y'] + np.cumsum(v_global[:,1] * dt) 
    z = initial_pos['z'] + np.cumsum(v_global[:,2] * dt)

    return t, x, y, z

# --------------------------------------
# 3. Визуализация результатов
# --------------------------------------
def plot_results(t, x, y, z, data):
    """Визуализация траектории и сравнение с GT"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D траектория
    ax3d = fig.add_subplot(211, projection='3d')
    ax3d.plot(x, y, z, label='Восстановленная траектория', linewidth=2)
    
    if all(k in data for k in ['x_gt', 'y_gt', 'z_gt']):
        ax3d.plot(data['x_gt'], data['y_gt'], data['z_gt'], 
                 label='GT', linestyle='--', alpha=0.7)
        
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()
    
    # Ошибки позиции
    if all(k in data for k in ['x_gt', 'y_gt', 'z_gt']):
        ax_err = fig.add_subplot(212)
        error = np.sqrt(
            (x - data['x_gt'])**2 + 
            (y - data['y_gt'])**2 + 
            (z - data['z_gt'])**2
        )
        ax_err.plot(t, error, color='red', label='Ошибка позиции')
        ax_err.set_xlabel('Время')
        ax_err.set_ylabel('Ошибка, м')
        ax_err.legend()
        plt.tight_layout()
        
        # Вывод метрик
        print(f"Средняя ошибка: {np.mean(error):.2f} м")
        print(f"Максимальная ошибка: {np.max(error):.2f} м")
        print(f"RMSE: {np.sqrt(mean_squared_error(zip(data['x_gt'], data['y_gt'], data['z_gt']), zip(x,y,z))):.2f} м")

    plt.show()

# --------------------------------------
# 4. Фильтрация результатов (опционально)
# --------------------------------------
def apply_filter(x, y, z, window=15, order=3):
    """Сглаживание траектории фильтром Савицкого-Голея"""
    return (
        savgol_filter(x, window, order),
        savgol_filter(y, window, order),
        savgol_filter(z, window, order)
    )

# --------------------------------------
# Главная функция
# --------------------------------------
if __name__ == "__main__":
    # Загрузка данных
    data = load_data("drone_state.txt")
    
    # Восстановление траектории
    t, x, y, z = reconstruct_trajectory(data)
    
    # Применение фильтра (опционально)
    # x, y, z = apply_filter(x, y, z)
    
    # Визуализация
    plot_results(t, x, y, z, data)