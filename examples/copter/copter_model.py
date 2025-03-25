#!/usr/bin/python3
import os
import math
import torch
import torch.nn as nn
# default NN model
import matplotlib.pyplot as plt
from robot_nn.defualt_nn_model import DefaultModel
import numpy as np


class QuadcopterModelLoss(nn.Module):
    def __init__(self):
        super(QuadcopterModelLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predict, ground_truth):
        """
        Defines the computation performed at every call.
        Args:
            predict (torch.tensor of shape [batch, time, state]): predicted trajectory (x, y, z, roll, pitch, yaw)
            ground_truth (torch.tensor of shape [batch, time, state]): ground truth trajectory (x, y, z, roll, pitch, yaw)
        Return:
            Calculated MSE loss for x, y, z coordinates
        """
        # Loss computed on position (x, y, z)
        return self.loss_fn(predict[:, :, 0:3], ground_truth[:, :, 0:3])

class QuadcopterModel(nn.Module):
    def __init__(
        self,
        n_layers=2,
        hidden_size=64,
        activation_function='elu',
        learning_rate=0.002,
        model_type="semilinear"
    ):
        """
        Args:
            :n_layers (int): number of layers in DefaultModel
            :hidden_size (int): number of neurons in hidden layers
            :activation_function (str): activation function (elu or relu)
            :learning_rate (float): neural network learning rate parameter
            :model_type (str): nonlinear / linear / semilinear
        Attributes:
            :model (torch.nn.Module): Dynamic quadcopter model based on a neural network
            :optim_lr: Neural network learning rate parameter
        """
        super(QuadcopterModel, self).__init__()

        self.model_type = model_type
        self.optim_lr = learning_rate

        # Input: [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed, servo1_raw, servo2_raw, servo3_raw, servo4_raw, dt]
        # Total input size: 6 (state) + 4 (control) + 1 (dt) = 11
        # Output: [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
        # Total output size: 6
        if self.model_type == "linear":
            # Для линейной модели: вход 11, выход 8 (6 для скоростей + 2 для коэффициентов)
            self.model = DefaultModel(
                n_inputs=11,  # V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed, servo1_raw, servo2_raw, servo3_raw, servo4_raw, dt
                n_outputs=8,  # 6 скоростей + 2 дополнительных выхода для линейной комбинации
                n_layers=n_layers,
                hidden_size=hidden_size,
                activation_function=activation_function
            )
        elif self.model_type == "semilinear":
            # Основная модель: вход 11, выход 6 (4 для коэффициентов + 2 для масштабирования)
            self.model = DefaultModel(
                n_inputs=11,
                n_outputs=6,  # 4 для alpha1 + 2 для alpha2
                n_layers=n_layers,
                hidden_size=hidden_size,
                activation_function=activation_function
            )
            # Дополнительная модель для bias: вход 11, выход 6
            self.b_model = DefaultModel(
                n_inputs=11,
                n_outputs=6,  # 6 для bias (V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed)
                n_layers=n_layers,
                hidden_size=hidden_size,
                activation_function=activation_function
            )
        elif self.model_type == "nonlinear":
            # Нелинейная модель: вход 11, выход 6
            self.model = DefaultModel(
                n_inputs=11,
                n_outputs=6,  # V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed
                n_layers=n_layers,
                hidden_size=hidden_size,
                activation_function=activation_function
            )

    def get_optimizer(self):
        """
        Creates and returns an optimizer for a neural network
        """
        return torch.optim.Adam(self.parameters(), lr=self.optim_lr)

    def get_loss_fn(self):
        """
        Creates and returns a loss function for a neural network
        """
        return QuadcopterModelLoss()

    def get_initial_state(self):
        """
        Creates and returns the initial state of the model
        """
        # Initial state: [x, y, z, roll, pitch, yaw, V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
        return torch.zeros([12])

    def update_state(self, state, control, dt=0.033, gt_velocities=None):
        """
        Args:
            :state (torch.tensor of shape [batch, 12]):
                batch of [x, y, z, roll, pitch, yaw, V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
            :control (torch.tensor of shape [batch, 4]): batch of [servo1_raw, servo2_raw, servo3_raw, servo4_raw]
            :dt (torch.tensor of shape [batch, 1] or float): time delta
        Return:
            :next_state (torch.tensor of shape [batch, 12]):
                batch of [x_next, y_next, z_next, roll_next, pitch_next, yaw_next, V_x_next, V_y_next, V_z_next, rollspeed_next, pitchspeed_next, yawspeed_next]
        """
        if isinstance(dt, float):
            dt = dt * torch.ones(state.shape[0], device=state.device)[:, None]

        # Extract current position and orientation
        x_curr, y_curr, z_curr = state[:, 0:1], state[:, 1:2], state[:, 2:3]  # shape [batch, 1]
        roll_curr, pitch_curr, yaw_curr = state[:, 3:4], state[:, 4:5], state[:, 5:6]  # shape [batch, 1]
        velocities_curr = state[:, 6:]  # shape [batch, 6] (V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed)

        # Prepare input for the model: [state velocities, control, dt]
        inp = torch.cat([velocities_curr, control, dt], dim=1)  # shape [batch, 11]

        # Predict new velocities
        if gt_velocities is None:
            predicted_velocities = self(inp)  # shape [batch, 6]
        else:
            predicted_velocities = gt_velocities

        # Extract predicted velocities
        V_x, V_y, V_z = predicted_velocities[:, 0:1], predicted_velocities[:, 1:2], predicted_velocities[:, 2:3]  # shape [batch, 1]
        rollspeed, pitchspeed, yawspeed = predicted_velocities[:, 3:4], predicted_velocities[:, 4:5], predicted_velocities[:, 5:6]  # shape [batch, 1]

        # Update orientation (roll, pitch, yaw)
        roll = roll_curr + rollspeed * dt  # shape [batch, 1]
        pitch = pitch_curr + pitchspeed * dt  # shape [batch, 1]
        yaw = yaw_curr + yawspeed * dt  # shape [batch, 1]

        # Normalize angles to [-pi, pi]
        for angle in [roll, pitch, yaw]:
            mask = (angle > math.pi) * (2 * math.pi)
            angle -= mask
            mask = (angle < -math.pi) * (2 * math.pi)
            angle += mask

        # Update position (x, y, z) using velocities
        x = x_curr + V_x * dt  # shape [batch, 1]
        y = y_curr + V_y * dt  # shape [batch, 1]
        z = z_curr + V_z * dt  # shape [batch, 1]

        # Combine into next state
        next_state = torch.cat([
            x, y, z, roll, pitch, yaw,
            predicted_velocities
        ], dim=1)  # shape [batch, 12]

        return next_state

    def forward(self, inp):
        """
        Defines the computation performed at every call.
        Args:
            :inp: (torch.tensor of shape [batch, 11]) input tensor
                  [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed, servo1_raw, servo2_raw, servo3_raw, servo4_raw, dt]
        Return:
            :output: (torch.tensor of shape [batch, 6]) predicted velocities
                     [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
        """
        if self.model_type == "linear":
            velocities = inp[:, :6]  # shape [batch, 6]
            servos = inp[:, 6:10]  # shape [batch, 4]
            alphas = self.model(inp**2)  # shape [batch, 8]
            alpha1 = torch.sigmoid(alphas[:, :6])  # shape [batch, 6] (for velocities)
            alpha2 = torch.nn.functional.elu(alphas[:, 6:8]) + 1  # shape [batch, 2] (for servos)
            beta = alphas[:, 6:8] * 0.01  # shape [batch, 2]
            # Linear combination: extend alpha2 and beta to match velocities dimension
            alpha2_extended = torch.cat([alpha2, alpha2, alpha2], dim=1)  # shape [batch, 6]
            beta_extended = torch.cat([beta, beta, beta], dim=1)  # shape [batch, 6]
            new_velocities = velocities * alpha1 + alpha2_extended + beta_extended
            return new_velocities  # shape [batch, 6]
        elif self.model_type == "semilinear":
            velocities = inp[:, :6]  # shape [batch, 6]
            servos = inp[:, 6:10]  # shape [batch, 4]
            alphas = self.model(inp**2)  # shape [batch, 6]
            alpha1 = torch.sigmoid(alphas[:, :4])  # shape [batch, 4] (for servos)
            alpha2 = torch.nn.functional.elu(alphas[:, 4:6]) + 1  # shape [batch, 2] (for scaling)
            beta = self.b_model(inp) * 0.01  # shape [batch, 6]
            # Combine servos with alpha1 and scale with alpha2
            servo_contrib = torch.cat([servos[:, :2] * alpha1[:2], servos[:, 2:4] * alpha1[2:]], dim=1)  # shape [batch, 4]
            servo_contrib = torch.cat([servo_contrib, servo_contrib[:, :2] * alpha2], dim=1)  # shape [batch, 6]
            new_velocities = velocities + servo_contrib + beta
            return new_velocities  # shape [batch, 6]
        elif self.model_type == "nonlinear":
            return self.model(inp)  # shape [batch, 6]

    def calc_metrics(self, predict, ground_truth):
        """
        Calls functions to calculate prediction metrics
        Args:
            :predict: (torch.tensor of shape [batch size, time, state])
                the trajectory predicted by the neural network (x, y, z, roll, pitch, yaw, ...)
            :ground_truth: (torch.tensor of shape [batch size, time, state])
                the ground truth trajectory (x, y, z, roll, pitch, yaw, ...)
        Return:
            :result: (dict) Dictionary with the results of calculating different metrics.
        """
        result = dict()
        result['traj_ate'] = self.calc_ate(predict, ground_truth)
        result['orientation_mae'] = self.calc_orientation_mae(predict, ground_truth)
        return result

    def calc_ate(self, predict, ground_truth):
        """
        Calculates the average translation error for given prediction
        Args:
            :predict: (torch.tensor of shape [batch size, time, state]) predicted trajectory
            :ground_truth: (torch.tensor of shape [batch size, time, state]) ground truth trajectory
        Return:
            :err: (torch.tensor of shape [batch size, 1]) calculated error
        """
        with torch.no_grad():
            mse_x = torch.square(predict[:, :, 0] - ground_truth[:, :, 0])  # shape [batch size, time]
            mse_y = torch.square(predict[:, :, 1] - ground_truth[:, :, 1])  # shape [batch size, time]
            mse_z = torch.square(predict[:, :, 2] - ground_truth[:, :, 2])  # shape [batch size, time]
            err = torch.mean(torch.sqrt(mse_x + mse_y + mse_z)).cpu().detach().numpy()
        return err

    def calc_orientation_mae(self, predict, ground_truth):
        """
        Calculates the mean absolute error for roll, pitch, and yaw
        Args:
            :predict: (torch.tensor of shape [batch size, time, state]) predicted trajectory
            :ground_truth: (torch.tensor of shape [batch size, time, state]) ground truth trajectory
        Return:
            :err: (torch.tensor of shape [batch size, 1]) calculated error
        """
        with torch.no_grad():
            err_roll = torch.mean(torch.abs(predict[:, :, 3] - ground_truth[:, :, 3]))
            err_pitch = torch.mean(torch.abs(predict[:, :, 4] - ground_truth[:, :, 4]))
            err_yaw = torch.mean(torch.abs(predict[:, :, 5] - ground_truth[:, :, 5]))
            err = (err_roll + err_pitch + err_yaw) / 3
        return err.cpu().detach().numpy()

    def plot_trajectories(self, predict, ground_truth_traj):
        """
        A helper function that takes the predicted and ground truth
        trajectory and plots them on the same graph.
        Args:
            :predict: (torch.tensor of shape [batch size, time, state])
                the trajectory predicted by the neural network
            :ground_truth_traj: (QuadcopterDataset) Single trajectory dataset
        Return:
            :fig: (matplotlib.figure.Figure) Several plots on one figure
        """
        ground_truth_pose = ground_truth_traj.data_pose.cpu().numpy()  # [x, y, z, roll, pitch, yaw]
        ground_truth_state = ground_truth_traj.data_x.cpu().numpy()  # [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
        control = ground_truth_traj.data_u.cpu().numpy()  # [servo1_raw, servo2_raw, servo3_raw, servo4_raw]
        time = ground_truth_traj.data_t.cpu().numpy()  # [time]

        predict = predict.cpu().numpy()  # [x, y, z, roll, pitch, yaw, V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]

        fig, ax = plt.subplots(9, figsize=(7, 30))  # Adjusted for more plots

        # Plot linear velocities (V_x, V_y, V_z)
        ax[0].set_ylabel('m/s')
        ax[0].set_title("Linear Velocities (V_x, V_y, V_z)")
        ax[0].plot(time[:, 0], ground_truth_state[:, 0], color='black', label='V_x', ls='--')
        ax[0].plot(time[:, 0], predict[:, 6], color='red', label='Predict V_x')
        ax[0].plot(time[:, 0], ground_truth_state[:, 1], color='blue', label='V_y', ls='--')
        ax[0].plot(time[:, 0], predict[:, 7], color='orange', label='Predict V_y')
        ax[0].plot(time[:, 0], ground_truth_state[:, 2], color='green', label='V_z', ls='--')
        ax[0].plot(time[:, 0], predict[:, 8], color='purple', label='Predict V_z')
        ax[0].legend(loc="lower right")

        # Plot angular velocities (rollspeed, pitchspeed, yawspeed)
        ax[1].set_ylabel('rad/s')
        ax[1].set_title("Angular Velocities (rollspeed, pitchspeed, yawspeed)")
        ax[1].plot(time[:, 0], ground_truth_state[:, 3], color='black', label='rollspeed', ls='--')
        ax[1].plot(time[:, 0], predict[:, 9], color='red', label='Predict rollspeed')
        ax[1].plot(time[:, 0], ground_truth_state[:, 4], color='blue', label='pitchspeed', ls='--')
        ax[1].plot(time[:, 0], predict[:, 10], color='orange', label='Predict pitchspeed')
        ax[1].plot(time[:, 0], ground_truth_state[:, 5], color='green', label='yawspeed', ls='--')
        ax[1].plot(time[:, 0], predict[:, 11], color='purple', label='Predict yawspeed')
        ax[1].legend(loc="lower right")

        # Plot control inputs (servo1_raw, servo2_raw, servo3_raw, servo4_raw)
        ax[2].set_ylabel('Servo Value')
        ax[2].set_title("Control Inputs (Servo Values)")
        ax[2].plot(time[:, 0], control[:, 0], color='black', label='servo1_raw')
        ax[2].plot(time[:, 0], control[:, 1], color='blue', label='servo2_raw')
        ax[2].plot(time[:, 0], control[:, 2], color='green', label='servo3_raw')
        ax[2].plot(time[:, 0], control[:, 3], color='purple', label='servo4_raw')
        ax[2].legend(loc="lower right")

        # Plot x coordinate over time
        ax[3].set_ylabel('m')
        ax[3].set_xlabel('t, sec')
        ax[3].set_title("X Coordinate over Time")
        ax[3].plot(time[:, 0], ground_truth_pose[:, 0], color='black', label='X(t)', ls='--')
        ax[3].plot(time[:, 0], predict[:, 0], color='red', label='Predict X(t)')
        ax[3].legend(loc="lower right")

        # Plot y coordinate over time
        ax[4].set_ylabel('m')
        ax[4].set_xlabel('t, sec')
        ax[4].set_title("Y Coordinate over Time")
        ax[4].plot(time[:, 0], ground_truth_pose[:, 1], color='black', label='Y(t)', ls='--')
        ax[4].plot(time[:, 0], predict[:, 1], color='red', label='Predict Y(t)')
        ax[4].legend(loc="lower right")

        # Plot z coordinate over time
        ax[5].set_ylabel('m')
        ax[5].set_xlabel('t, sec')
        ax[5].set_title("Z Coordinate over Time")
        ax[5].plot(time[:, 0], ground_truth_pose[:, 2], color='black', label='Z(t)', ls='--')
        ax[5].plot(time[:, 0], predict[:, 2], color='red', label='Predict Z(t)')
        ax[5].legend(loc="lower right")

        # Plot roll, pitch, yaw over time
        ax[6].set_ylabel('rad')
        ax[6].set_xlabel('t, sec')
        ax[6].set_title("Orientation (roll, pitch, yaw) over Time")
        ax[6].plot(time[:, 0], ground_truth_pose[:, 3], color='black', label='roll(t)', ls='--')
        ax[6].plot(time[:, 0], predict[:, 3], color='red', label='Predict roll(t)')
        ax[6].plot(time[:, 0], ground_truth_pose[:, 4], color='blue', label='pitch(t)', ls='--')
        ax[6].plot(time[:, 0], predict[:, 4], color='orange', label='Predict pitch(t)')
        ax[6].plot(time[:, 0], ground_truth_pose[:, 5], color='green', label='yaw(t)', ls='--')
        ax[6].plot(time[:, 0], predict[:, 5], color='purple', label='Predict yaw(t)')
        ax[6].legend(loc="lower right")

        # Plot XY trajectory
        ax[7].set_ylabel('Y, m')
        ax[7].set_xlabel('X, m')
        ax[7].set_title("XY Trajectory")
        ax[7].plot(ground_truth_pose[:, 0], ground_truth_pose[:, 1], color='black', label='XY', ls='--')
        ax[7].plot(predict[:, 0], predict[:, 1], color='red', label='Predict XY')
        ax[7].legend(loc="lower right")

        # Plot XZ trajectory
        ax[8].set_ylabel('Z, m')
        ax[8].set_xlabel('X, m')
        ax[8].set_title("XZ Trajectory")
        ax[8].plot(ground_truth_pose[:, 0], ground_truth_pose[:, 2], color='black', label='XZ', ls='--')
        ax[8].plot(predict[:, 0], predict[:, 2], color='red', label='Predict XZ')
        ax[8].legend(loc="lower right")

        plt.tight_layout()
        return fig

    def save_predict_to_csv(self, predict, ground_truth_traj, path):
        """
        Stores neural network prediction and ground truth data in csv format.
        Args:
            :predict: (torch.tensor of shape [batch size, num_samples, state])
                the trajectory predicted by the neural network
            :ground_truth_traj: (QuadcopterDataset) Single trajectory dataset
            :path: (str) file path
        """
        ground_truth_pose = ground_truth_traj.data_pose.cpu().detach().numpy()  # [x, y, z, roll, pitch, yaw]
        ground_truth_state = ground_truth_traj.data_x.cpu().detach().numpy()  # [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
        control = ground_truth_traj.data_u.cpu().detach().numpy()  # [servo1_raw, servo2_raw, servo3_raw, servo4_raw]
        time_seq = ground_truth_traj.data_t.cpu().detach().numpy()  # [time]

        predict = predict.cpu().detach().numpy()  # [x, y, z, roll, pitch, yaw, V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]

        if not os.path.exists(path):
            os.makedirs(path)

        # Save all data in a single CSV file, matching the QuadcopterDataset format
        combined_data = np.hstack([
            time_seq,  # Time
            ground_truth_pose,  # x, y, z, roll, pitch, yaw
            ground_truth_state,  # V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed
            control  # servo1_raw, servo2_raw, servo3_raw, servo4_raw
        ])
        combined_predict = np.hstack([
            time_seq,  # Time
            predict[:, :6],  # x, y, z, roll, pitch, yaw
            predict[:, 6:],  # V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed
            control  # servo1_raw, servo2_raw, servo3_raw, servo4_raw
        ])

        header = 'Time,x,y,z,roll,pitch,yaw,V_x,V_y,V_z,rollspeed,pitchspeed,yawspeed,servo1_raw,servo2_raw,servo3_raw,servo4_raw'
        np.savetxt(
            os.path.join(path, "ground_truth.csv"),
            combined_data,
            delimiter=',',
            header=header,
            comments=''
        )
        np.savetxt(
            os.path.join(path, "nn_model_predict.csv"),
            combined_predict,
            delimiter=',',
            header=header,
            comments=''
        )