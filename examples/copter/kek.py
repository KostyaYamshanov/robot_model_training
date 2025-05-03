import numpy as np
import matplotlib.pyplot as plt
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

# --------------------------------------
# 2. Класс состояния дрона
# --------------------------------------
class DroneState:
    def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, roll=0.0, pitch=0.0, yaw=0.0):

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Создание матрицы поворота из углов Эйлера (ZYX порядок)"""
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        return np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,            cp*sr,            cp*cr]
        ])

    #def update_state(self, throttle, roll_sp, pitch_sp, steering, g=9.81, dt=0.05, mass=0.8, max_thrust=20.0, max_angle=np.radians(30)):
    #def update_state(self, throttle, roll_sp, pitch_sp, yaw_rate, g=9.81, vx=0.0, vy=0.0, vz=0.0, dt=0.05, mass=0.8, max_thrust=16.7, max_angle=np.radians(30)):
    def update_state(self, throttle, roll_sp, pitch_sp, yaw_rate, g=9.81, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0, dt=0.05, mass=0.8, max_thrust=16.7, max_angle=np.radians(30)):

        """Обновление состояния с нормализацией throttle"""

        roll = self.roll + wx * dt 
        pitch = self.pitch + wy * dt
        yaw = self.yaw + yaw_rate * dt


        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        if self.z < 0.0:
            self.z = 0.0

        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

# --------------------------------------
# 3. Восстановление траектории
# --------------------------------------
def reconstruct_trajectory(data, control_data):
    """Восстановление траектории"""
    required_keys = ['roll', 'pitch', 'yaw']
    control_keys = ['roll', 'pitch', 'throttle', 'steering']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Недостает данных в drone_state: {key}")
    for key in control_keys:
        if key not in control_data:
            raise KeyError(f"Недостает данных в control: {key}")

    state = DroneState(
        x=0.0, y=0.0, z=0.0,
        roll=data['roll'][0],
        pitch=data['pitch'][0],
        yaw=data['yaw'][0]
    )
    
    n = min(len(control_data['throttle']), len(control_data['roll']), len(control_data['pitch']), len(data['vx']))
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    dt = 0.049

    for i in range(n):
        x[i] = state.x
        y[i] = state.y
        z[i] = state.z
        
        yaw_rate = control_data['steering'][i]
        state.update_state(
            throttle=control_data['throttle'][i],
            roll_sp=control_data['roll'][i],
            pitch_sp=control_data['pitch'][i],
            yaw_rate=yaw_rate,
            vx=data['vx'][i],
            vy=data['vz'][i],
            vz=data['vy'][i],
            wx=data['wx'][i],
            wy=data['wy'][i],
            wz=data['wz'][i],
            dt=dt
        )
    
    return dt * np.arange(n), x, y, z

# --------------------------------------
# 4. Визуализация
# --------------------------------------
def plot_comparison(t, x, y, z, data):
    """Визуализация результатов с отдельными графиками по осям"""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
    
    ax3d = fig.add_subplot(gs[0], projection='3d')
    
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
        rmse_3d = np.sqrt(mean_squared_error(np.vstack([gt_x, gt_y, gt_z]).T, 
                                           np.vstack([x, y, z]).T))
    
    ax3d.plot(x, y, z, label='Восстановленная', linewidth=2)
    if has_gt:
        ax3d.plot(gt_x, gt_y, gt_z, linestyle='--', alpha=0.7, label='GT')
        ax3d.set_title(f'3D Траектория (RMSE: {rmse_3d:.2f} м)')
    ax3d.legend()
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    gs_low = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1])
    ax_x = fig.add_subplot(gs_low[0])
    ax_y = fig.add_subplot(gs_low[1])
    ax_z = fig.add_subplot(gs_low[2])
    
    ax_x.plot(t, x, label='Восстановленная')
    if has_gt:
        ax_x.plot(t, gt_x, linestyle='--', label='GT')
        rmse_x = np.sqrt(mean_squared_error(gt_x, x))
        ax_x.set_title(f'X(t) (RMSE: {rmse_x:.2f} м)')
    ax_x.set_xlabel('Время, с')
    ax_x.set_ylabel('X, м')
    ax_x.legend()
    
    ax_y.plot(t, y, label='Восстановленная')
    if has_gt:
        ax_y.plot(t, gt_y, linestyle='--', label='GT')
        rmse_y = np.sqrt(mean_squared_error(gt_y, y))
        ax_y.set_title(f'Y(t) (RMSE: {rmse_y:.2f} м)')
    ax_y.set_xlabel('Время, с')
    ax_y.set_ylabel('Y, м')
    ax_y.legend()
    
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

# --------------------------------------
# 5. Запуск
# --------------------------------------
if __name__ == "__main__":
    # Загрузка данных
    data = load_data("drone_state.txt")
    control_data = load_data("control.txt")
    
    # Проверка ключей и длины данных
    print("Ключи в drone_state.txt:", data.keys())
    print("Ключи в control.txt:", control_data.keys())
    
    # Проверка диапазона throttle
    throttle_raw = control_data['throttle']
    print("Throttle пример (первые 5):", throttle_raw[:5])
    min_throttle = min(throttle_raw)
    max_throttle = max(throttle_raw)
    print("Throttle диапазон: min =", min_throttle, "max =", max_throttle)
    
    # Нормализация throttle (если не встроена в update_state)
    # Если диапазон не [-1, 1], раскомментируйте и настройте
    control_data['throttle'] = [(t - min_throttle) / (max_throttle - min_throttle) for t in throttle_raw]
    
    # Проверка диапазона углов
    print("Roll пример (первые 5):", control_data['roll'][:5])
    print("Pitch пример (первые 5):", control_data['pitch'][:5])
    print("Steering пример (первые 5):", control_data['steering'][:5])
    
    # Проверка длины данных
    n_state = min(len(data['roll']), len(data['pitch']), len(data['yaw']))
    n_control = min(len(control_data['throttle']), len(control_data['roll']), 
                   len(control_data['pitch']), len(control_data['steering']))
    print(f"Длина данных: drone_state = {n_state}, control = {n_control}")
    
    # Реконструкция траектории
    try:
        t, x, y, z = reconstruct_trajectory(data, control_data)
        print(f"Траектория восстановлена: {len(t)} точек")
        
        # Вывод первых нескольких точек для проверки
        print("Первые 5 точек восстановленной траектории:")
        for i in range(min(5, len(t))):
            print(f"t={t[i]:.3f}, x={x[i]:.3f}, y={y[i]:.3f}, z={z[i]:.3f}")
        
        # Визуализация
        plot_comparison(t, x, y, z, data)
    except KeyError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка при реконструкции: {e}")