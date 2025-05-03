import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec
from pyswarm import pso

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

    # def update_state(self, u4, g=9.80665, dt=0.05):
    #     """Обновление состояния через уравнения динамики"""

    #     # u1, u2, u3 = self.roll, self.pitch, self.yaw
    #     # # Вычисление ускорений
    #     # ax = u4 * (np.sin(u3) * np.cos(u2) * np.cos(u1) + np.sin(u1) * np.sin(u2))
    #     # ay = u4 * np.cos(u3) * np.cos(u1) - g 
    #     # az = u4 * (np.cos(u2) * np.sin(u1) - np.cos(u1) * np.sin(u2) * np.sin(u3)) 
        
    #     # Ускорение в глобальной системе координат
    #     rotation = Rotation.from_euler('ZYX', [self.yaw, self.pitch, self.roll])
    #     R = rotation.as_matrix()
    #     ax = u4 * R[0, 2]
    #     ay = u4 * R[1, 2] - g 
    #     az = u4 * R[2, 2]

    #     # Интегрирование
    #     self.vx += ax * dt
    #     self.vy += ay * dt
    #     self.vz += az * dt
    #     self.x += self.vx * dt
    #     self.y += self.vy * dt
    #     self.z += self.vz * dt

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Создание матрицы поворота из углов Эйлера (ZYX порядок), как в симуляторе"""
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,            cp*sr,            cp*cr]
        ])
        return R

    def update_state(self, u4, roll_sp, pitch_sp, yaw_rate, g=9.81, dt=0.02, mass=0.5, max_thrust=17.0, max_angle=np.radians(30)):
        """Обновление состояния на основе динамики симулятора"""
        # Ограничение управления
        # thrust = np.clip(u4, 0.0, 1.0) * max_thrust
        thrust_normalized = (u4 + 1) / 2
        thrust = np.clip(thrust_normalized, 0.0, 1.0) * max_thrust
        roll_sp = np.clip(roll_sp, -max_angle, max_angle)
        pitch_sp = np.clip(pitch_sp, -max_angle, max_angle)

        # Смешивание заданных углов (как в симуляторе)
        # roll = roll_sp  # Можно добавить веса, как в симуляторе: 0.7 * roll_sp + 0.3 * self.roll
        # pitch = pitch_sp

        yaw = self.yaw + yaw_rate * dt
        roll = 0.7 * roll_sp + 0.3 * self.roll
        pitch = 0.7 * pitch_sp + 0.3 * self.pitch
        # Матрица поворота
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # Уравнения движения
        thrust_global = R @ np.array([0.0, 0.0, thrust])
        self.acceleration = (thrust_global - np.array([0.0, 0.0, mass * g])) / mass
        
        # Интегрирование
        self.vx += self.acceleration[0] * dt
        self.vy += self.acceleration[1] * dt
        self.vz += self.acceleration[2] * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Ограничение высоты
        if self.z < 0.0:
            self.z = 0.0

        # Обновление углов
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

def objective_function(params, data):
    mass, max_thrust = params
    rmse, _, _, _, _ = reconstruct_trajectory(data, mass, max_thrust)
    return rmse

def reconstruct_trajectory(data, mass, max_thrust):
    """Восстановление траектории с заданными mass и max_thrust"""
    required_keys = ['roll', 'pitch', 'yaw', 'u4']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Недостает данных: {key}")
    
    
    state = DroneState(
        x=0.0, y=0.0, z=0.0,
        roll=data['roll'][0],
        pitch=data['pitch'][0],
        yaw=data['yaw'][0]
    )
    
    n = len(data['u4']) - 1
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    dt = 0.05  # Фиксированный шаг из вашего кода
    
    for i in range(n):
        x[i] = state.x
        y[i] = state.y
        z[i] = state.z
        
        state.update_state(
            data['u4'][i],
            data['roll'][i],
            data['pitch'][i],
            data['yaw'][i],  # Используем yaw как yaw_rate для простоты
            dt=dt,
            mass=mass,
            max_thrust=max_thrust
        )
    
    # Вычисляем RMSE, если есть реальные данные
    if all(k in data for k in ['x', 'y', 'z']):
        min_len = min(len(data['x']), len(data['y']), len(data['z']), n)
        gt_x = np.array(data['x'][:min_len])
        gt_y = np.array(data['y'][:min_len])
        gt_z = np.array(data['z'][:min_len])
        rec_x = x[:min_len]
        rec_y = y[:min_len]
        rec_z = z[:min_len]
        rmse = np.sqrt(mean_squared_error(
            np.vstack([gt_x, gt_y, gt_z]).T,
            np.vstack([rec_x, rec_y, rec_z]).T
        ))
        return rmse, dt * np.arange(n), x, y, z  # Возвращаем RMSE и траекторию
    else:
        raise ValueError("Реальные данные о положении (x, y, z) отсутствуют")


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


# После загрузки данных в if __name__ == "__main__":
if __name__ == "__main__":
    # Загрузка данных
    data = load_data("drone_state.txt")
    print(f"data.keys = {data.keys()}")
    control_data = load_data("control.txt")
    print(f"control_data.keys = {control_data.keys()}")
    
    # Нормализация throttle (как в вашем коде)
    control_data['throttle'] = [(-1.0 * t + 1) / 2 for t in control_data['throttle']]
    MASS_K = 20  # Из вашего кода
    data['u4'] = [t / MASS_K for t in control_data['throttle']]
    
    # Границы для параметров
    lb = [0.5, 5.0]  # Нижние границы: mass (кг), max_thrust (Н)
    ub = [1.5, 20.0] # Верхние границы: mass (кг), max_thrust (Н)
    
    # Запуск PSO
    optimal_params, min_rmse = pso(
        objective_function,
        lb,
        ub,
        args=(data,),
        swarmsize=100,  # Количество частиц
        maxiter=1000     # Максимальное число итераций
    )
    
    mass_opt, max_thrust_opt = optimal_params
    print(f"Оптимальные параметры: mass={mass_opt:.2f} кг, max_thrust={max_thrust_opt:.2f} Н")
    print(f"Минимальный RMSE: {min_rmse:.2f} м")
    
    # Восстановление траектории с оптимальными параметрами
    _, t, x, y, z = reconstruct_trajectory(data, mass_opt, max_thrust_opt)
    
    # Визуализация
    plot_comparison(t, x, y, z, data)