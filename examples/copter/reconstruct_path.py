import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec

# --------------------------------------
# 1. Загрузка данных
# --------------------------------------
def load_data(file_path):
    """Загрузка данных из файла с проверкой типов"""
    import ast
    with open(file_path, "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
    
    return data

class DroneState:
    def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        # Явное преобразование к float
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def update_state(self, u4, g=9.80665, dt=0.05):
        """Обновление состояния через уравнения динамики"""

        # u1, u2, u3 = self.roll, self.pitch, self.yaw
        # # Вычисление ускорений
        # ax = u4 * (np.sin(u3) * np.cos(u2) * np.cos(u1) + np.sin(u1) * np.sin(u2))
        # ay = u4 * np.cos(u3) * np.cos(u1) - g 
        # az = u4 * (np.cos(u2) * np.sin(u1) - np.cos(u1) * np.sin(u2) * np.sin(u3)) 
        
        # Ускорение в глобальной системе координат
        rotation = Rotation.from_euler('ZYX', [self.yaw, self.pitch, self.roll])
        R = rotation.as_matrix()
        ax = u4 * R[0, 2]
        ay = u4 * R[1, 2] - g 
        az = u4 * R[2, 2]

        # Интегрирование
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

def reconstruct_trajectory(data):
    """Восстановление траектории"""
    required_keys = ['roll', 'pitch', 'yaw', 'u4']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Недостает данных: {key}")
    
    # Инициализация состояния
    state = DroneState(
        x=0.0,
        y=0.0,
        z=0.0,
        roll=data['roll'][0],
        pitch=data['pitch'][0],
        yaw=data['yaw'][0]
    )
    
    n = len(data['u4']) - 1
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    dt = 0.05  # Фиксированный шаг по времени
    
    roll_len = len(data['roll'])
    pitch_len = len(data['pitch'])
    yaw_len = len(data['yaw'])
    x_len = len(data['x'])
    print(f"n = {n}, len(roll) = {roll_len} len(pitch) = {pitch_len} len(yaw) = {yaw_len} len(x) = {x_len}")
    for i in range(n):
        x[i] = state.x
        y[i] = state.y
        z[i] = state.z
        
        # Обновляем углы и u4
        state.roll = data['roll'][i]
        state.pitch = data['pitch'][i]
        state.yaw = data['yaw'][i]
        # state.roll = np.radians(data['roll'][i])
        # state.pitch = np.radians(data['pitch'][i])
        # state.yaw = np.radians(data['yaw'][i])
        state.update_state(u4=data['u4'][i], dt=dt)
    
    return dt * np.arange(n), x, y, z


def plot_comparison(t, x, y, z, data):
    """Визуализация результатов с отдельными графиками по осям"""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
    
    # 3D траектория
    ax3d = fig.add_subplot(gs[0], projection='3d')
    
    # Проверка наличия GT данных и обрезка до одинаковой длины
    has_gt = all(k in data for k in ['x', 'y', 'z'])
    gt_x, gt_y, gt_z = [], [], []
    if has_gt:
        min_len = min(len(data['x']), len(data['y']), len(data['z']), len(x), len(y), len(z))
        gt_x = np.array(data['x'][:min_len])
        gt_y = np.array(data['y'][:min_len])
        gt_z = np.array(data['z'][:min_len])
        t = t[:min_len]
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        # Расчёт RMSE для 3D
        rmse_3d = np.sqrt(mean_squared_error(np.vstack([gt_x, gt_y, gt_z]).T, 
                                           np.vstack([x, y, z]).T))
    
    # Построение 3D траектории
    ax3d.plot(x, y, z, label='Восстановленная', linewidth=2)
    if has_gt:
        ax3d.plot(gt_x, gt_y, gt_z, linestyle='--', alpha=0.7, label='GT')
        ax3d.set_title(f'3D Траектория (RMSE: {rmse_3d:.2f} м)')
    ax3d.legend()
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    # Графики по осям X, Y, Z
    gs_low = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1])
    ax_x = fig.add_subplot(gs_low[0])
    ax_y = fig.add_subplot(gs_low[1])
    ax_z = fig.add_subplot(gs_low[2])
    
    # X(t)
    ax_x.plot(t, x, label='Восстановленная')
    if has_gt:
        ax_x.plot(t, gt_x, linestyle='--', label='GT')
        rmse_x = np.sqrt(mean_squared_error(gt_x, x))
        ax_x.set_title(f'X(t) (RMSE: {rmse_x:.2f} м)')
    ax_x.set_xlabel('Время, с')
    ax_x.set_ylabel('X, м')
    ax_x.legend()
    
    # Y(t)
    ax_y.plot(t, y, label='Восстановленная')
    if has_gt:
        ax_y.plot(t, gt_y, linestyle='--', label='GT')
        rmse_y = np.sqrt(mean_squared_error(gt_y, y))
        ax_y.set_title(f'Y(t) (RMSE: {rmse_y:.2f} м)')
    ax_y.set_xlabel('Время, с')
    ax_y.set_ylabel('Y, м')
    ax_y.legend()
    
    # Z(t)
    ax_z.plot(t, z, label='Восстановленная')
    if has_gt:
        ax_z.plot(t, gt_z, linestyle='--', label='GT')
        rmse_z = np.sqrt(mean_squared_error(gt_z, z))
        ax_z.set_title(f'Z(t) (RMSE: {rmse_z:.2f} м)')
    ax_z.set_xlabel('Время, с')
    ax_z.set_ylabel('Z, м')
    ax_z.legend()
    
    plt.tight_layout()
    plt.show()

# Запуск реконструкции
if __name__ == "__main__":
    # Загрузка данных (пример)
    data = load_data("drone_state.txt")
    control_data = load_data("control.txt")
    MASS_K = 20 # mass * koeff kg
    data['u4'] = [t / MASS_K for t in control_data['throttle']]
    # Реконструкция траектории
    t, x, y, z = reconstruct_trajectory(data)
    
    # Визуализация
    plot_comparison(t, x, y, z, data)