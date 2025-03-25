#!/usr/bin/python3
# -*- coding: utf-8 -*-
# licence removed for brevity
import os
import torch
import pandas as pd
import numpy as np

class QuadcopterDataset:
    """
    Dataset class for a quadcopter.
    All data (state, control, time) is stored in a single CSV file.
    """
    def __init__(
        self,
        data_path,
        device='cpu'
    ):
        """
        Args:
            :data_path: (str) Path to the CSV file containing all quadcopter data
            :device: (str) 'cuda' or 'cpu'
        Attributes:
            :data_t: (torch.tensor of shape [num_samples, 1]) timestamp sequence
            :data_x: (torch.tensor of shape [num_samples, 6]) state sequence (V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed)
            :data_u: (torch.tensor of shape [num_samples, 4]) control sequence (servo1_raw, servo2_raw, servo3_raw, servo4_raw)
            :data_pose: (torch.tensor of shape [num_samples, 6]) pose sequence (x, y, z, roll, pitch, yaw)
        """
        super().__init__()
        
        # Parse the CSV file
        self.data_t, self.data_x, self.data_u, self.data_pose = self.parse_data(data_path)
        self.device = device

        # Move tensors to the specified device
        self.data_t = self.data_t.to(device)
        self.data_x = self.data_x.to(device)
        self.data_u = self.data_u.to(device)
        self.data_pose = self.data_pose.to(device)
    
    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.data_t)

    def parse_data(self, data_path):
        """
        Parse the CSV file containing all quadcopter data.
        Expected header: Time,x,y,z,roll,pitch,yaw,V_x,V_y,V_z,rollspeed,pitchspeed,yawspeed,servo1_raw,servo2_raw,servo3_raw,servo4_raw
        Args:
            :data_path: (str) Path to the CSV file
        Returns:
            :data_t: (torch.tensor of shape [num_samples, 1]) Time
            :data_x: (torch.tensor of shape [num_samples, 6]) State (V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed)
            :data_u: (torch.tensor of shape [num_samples, 4]) Control (servo1_raw, servo2_raw, servo3_raw, servo4_raw)
            :data_pose: (torch.tensor of shape [num_samples, 6]) Pose (x, y, z, roll, pitch, yaw)
        """
        # Read the CSV file with comma as the delimiter
        data = pd.read_csv(data_path, delimiter=',')


        print("Columns in CSV:", data.columns.tolist())  # Проверяем столбцы
        print("Number of rows:", len(data))  # Проверяем количество строк

        # Extract time
        data_t = data['Time'].values.reshape(-1, 1)  # Shape: [num_samples, 1]

        # Extract pose (x, y, z, roll, pitch, yaw)
        data_pose = data[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].values  # Shape: [num_samples, 6]

        # Extract state (V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed)
        data_x = data[['V_x', 'V_y', 'V_z', 'rollspeed', 'pitchspeed', 'yawspeed']].values  # Shape: [num_samples, 6]

        # Extract control (servo1_raw, servo2_raw, servo3_raw, servo4_raw)
        data_u = data[['servo1_raw', 'servo2_raw', 'servo3_raw', 'servo4_raw']].values  # Shape: [num_samples, 4]

        # Convert to torch tensors
        data_t = torch.tensor(data_t, dtype=torch.float)
        data_pose = torch.tensor(data_pose, dtype=torch.float)
        data_x = torch.tensor(data_x, dtype=torch.float)
        data_u = torch.tensor(data_u, dtype=torch.float)

        return data_t, data_x, data_u, data_pose

def QuadcopterDataset_test():
    """
    Dummy tests for QuadcopterDataset
    """
    try:
        # Test with a single dataset
        data_path = "/path/to/quadcopter/data.csv"  # Replace with actual path
        test_dataset = QuadcopterDataset(data_path=data_path)
        print("1 OK")
        print(f"Dataset size: {len(test_dataset)}")
        print(f"Time shape: {test_dataset.data_t.shape}")
        print(f"State shape: {test_dataset.data_x.shape}")
        print(f"Control shape: {test_dataset.data_u.shape}")
        print(f"Pose shape: {test_dataset.data_pose.shape}")
    except Exception as e:
        print("1 FAIL")
        raise e

    try:
        # Test with multiple datasets in a directory
        global_path = "/path/to/quadcopter/datasets/"  # Replace with actual path
        list_of_datasets = []
        for traj_data in os.listdir(global_path):
            data_path = os.path.join(global_path, traj_data)
            if os.path.isfile(data_path) and data_path.endswith('.csv'):
                list_of_datasets.append(QuadcopterDataset(data_path=data_path))
        if len(list_of_datasets) == len([f for f in os.listdir(global_path) if f.endswith('.csv')]):
            print("2 OK")
    except Exception as e:
        print("2 FAIL")
        raise e

def main():
    """
    Main function to run tests
    """
    QuadcopterDataset_test()

if __name__ == "__main__":
    main()