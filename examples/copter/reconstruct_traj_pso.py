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
        # Инициализация состояния дрона
        self.x, self.y, self.z = map(float, (x, y, z))
        self.vx, self.vy, self.vz = map(float, (vx, vy, vz))
        self.roll, self.pitch, self.yaw = map(float, (roll, pitch, yaw))

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Матрица поворота из Эйлера (ZYX)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,            cp*sr,            cp*cr]
        ])

    def update_state(self, u4, roll_sp, pitch_sp, yaw_rate, g=9.81, dt=0.02, mass=0.5, max_thrust=17.0, max_angle=np.radians(30)):
        """Обновление состояния: thrust u4 в [0,1]"""
        # Прямое масштабирование топлива
        thrust = np.clip(u4, -1.0, 1.0) * max_thrust
        thrust = (u4+1)/2 * max_thrust  # Ньютонов
        roll_sp = np.clip(roll_sp, -max_angle, max_angle)
        pitch_sp = np.clip(pitch_sp, -max_angle, max_angle)

        # Обновление углов
        yaw = self.yaw + yaw_rate * dt
        roll = 0.7 * roll_sp + 0.3 * self.roll
        pitch = 0.7 * pitch_sp + 0.3 * self.pitch
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # Силы и ускорение
        #  TEST 
        thrust_global = R @ np.array([0.0, 0.0, thrust / mass - g])
        # thrust_global = R @ np.array([0.0, 0.0, thrust ])
        gravity = np.array([0.0, 0.0, mass * g])
        # self.acceleration = (thrust_global - gravity) / mass
        self.acceleration = R @ np.array([0, 0, thrust / mass - g])

        # Интегрирование
        self.vx += self.acceleration[0] * dt
        self.vy += self.acceleration[1] * dt
        self.vz += self.acceleration[2] * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z = max(0.0, self.z + self.vz * dt)

        self.roll, self.pitch, self.yaw = roll, pitch, yaw


def reconstruct_trajectory(data, mass, max_thrust, dt=0.05):
    """Восстановление траектории по mass и max_thrust"""
    for key in ('roll','pitch','yaw','u4'):
        if key not in data: raise KeyError(f"Missing '{key}' in data")
    state = DroneState(roll=data['roll'][0], pitch=data['pitch'][0], yaw=data['yaw'][0])
    n = len(data['u4'])-1
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        x[i], y[i], z[i] = state.x, state.y, state.z
        # предполагаем data['yaw'] как угловую скорость
        state.update_state(data['u4'][i], data['roll'][i], data['pitch'][i], data['yaw'][i], dt=dt, mass=mass, max_thrust=max_thrust)

    if all(k in data for k in ('x','y','z')):
        gt = np.vstack([data['x'], data['y'], data['z']]).T
        rec = np.vstack([x, y, z]).T
        rmse = np.sqrt(mean_squared_error(gt, rec))
        return rmse, np.arange(n)*dt, x, y, z
    else:
        raise ValueError("Ground-truth positions missing.")


def objective_function(params, data):
    mass, max_thrust = params
    rmse, *_ = reconstruct_trajectory(data, mass, max_thrust)
    return rmse


def plot_comparison(t, x, y, z, data):
    """Визуализация результатов с отдельными графиками по осям"""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
    
    # 3D траектория
    ax3d = fig.add_subplot(gs[0], projection='3d')
    has_gt = all(k in data for k in ['x', 'y', 'z'])
    if has_gt:
        min_len = min(len(data['x']), len(data['y']), len(data['z']), len(x), len(y), len(z))
        gt_x = np.array(data['x'][:min_len])
        gt_y = np.array(data['y'][:min_len])
        gt_z = np.array(data['z'][:min_len])
        t = t[:min_len]
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        rmse_3d = np.sqrt(mean_squared_error(np.vstack([gt_x, gt_y, gt_z]).T, np.vstack([x, y, z]).T))
    
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


if __name__ == '__main__':
    data = load_data('drone_state.txt')
    ctrl = load_data('control.txt')
    ctrl['throttle'] = [(1 - t)/2 for t in ctrl['throttle']]
    MASS_K = 1
    data['u4'] = [t/MASS_K for t in ctrl['throttle']]

    lb, ub = [0.1,5.0], [2.0,20.0]
    opt, rmse = pso(objective_function, lb, ub, args=(data,), swarmsize=100, maxiter=100)
    print(f"mass={opt[0]:.2f}, thrust={opt[1]:.2f}, RMSE={rmse:.2f}")
    _, t, x, y, z = reconstruct_trajectory(data, *opt)
    plot_comparison(t, x, y, z, data)
