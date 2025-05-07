#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import yaml
import wandb
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import numpy as np

from copter_model import QuadcopterModel 
from copter_dataset import QuadcopterDataset 
from robot_nn.trainer import Trainer
from robot_nn.utils import parse_config, init_wandb

"""
launch example:

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="2" 
./quadcopter_train.py -cfg /path/to/quadcopter/train_configs/quadcopter_1.yaml

"""

def parse_args():
    """
    Parse arguments from the command line
    Return:
        args: (dict) contains command line arguments
    """
    args = {}
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', action='store', dest='cfg_file',
                        required=False, help='Yaml config for training')
    parser.add_argument('-name', action='store', dest='wandb_name',
                        required=False, help='Run name for wandb', default=None)
    cli_args = parser.parse_args()
    args['cfg_file'] = cli_args.cfg_file
    args['wandb_name'] = cli_args.wandb_name
    return args

def parse_datasets(path):
    list_of_datasets = list()
    for traj_data in os.listdir(path):
        data_path = os.path.join(path, traj_data)
        if os.path.isfile(data_path) and data_path.endswith('.csv'):
            dataset = QuadcopterDataset(data_path=data_path)
            print(f"Dataset {data_path}:")
            print(f"  data_pose shape: {dataset.data_pose.shape}")  # Должно быть [N, 6]
            print(f"  data_x shape: {dataset.data_x.shape}")  # Должно быть [N, 6]
            print(f"  data_u shape: {dataset.data_u.shape}")  # Должно быть [N, 4]
            print(f"  data_t shape: {dataset.data_t.shape}")  # Должно быть [N, 1]
            list_of_datasets.append(dataset)
    return list_of_datasets

def parse_all_datasets(path, exclude=[]):
    """
    Parses all dataset directories, saves each to a list
    Args:
        :path: (str) path to directory with data files
    Return:
        :list_of_datasets: (list) list of QuadcopterDatasets
    """
    list_of_datasets = list()
    for traj_type in os.listdir(path):
        if traj_type not in exclude:
            list_of_datasets += parse_datasets(os.path.join(path, traj_type))
    return list_of_datasets

def plot_vel_and_ctrl_distribution(list_of_datasets: list):
    """
    Args:
        :list_of_datasets: list of QuadcopterDataset
    Return:
        :fig: (matplotlib.figure.Figure) velocities and control distribution graph
    """
    with torch.no_grad():
        all_state = torch.tensor([])
        all_control = torch.tensor([])
        for dataset in list_of_datasets:
            all_state = torch.cat([all_state, dataset.data_x.cpu()])  # [V_x, V_y, V_z, rollspeed, pitchspeed, yawspeed]
            all_control = torch.cat([all_control, dataset.data_u.cpu()])  # [servo1_raw, servo2_raw, servo3_raw, servo4_raw]
        data_for_visualization = torch.cat([all_state, all_control], dim=1).numpy() 
        data_for_visualization = pd.DataFrame(
            data_for_visualization,
            columns=["V_x", "V_y", "V_z", "rollspeed", "pitchspeed", "yawspeed", 
                     "servo1_raw", "servo2_raw", "servo3_raw", "servo4_raw"]
        )
    fig, ax = plt.subplots()
    graph = sns.pairplot(data_for_visualization[::100], plot_kws={'alpha': 0.1})
    return fig

def main():
    """
    Main function to train and evaluate the QuadcopterModel
    """
    # Parse args
    args = parse_args()
    # Parse config
    config = parse_config(args["cfg_file"])
    use_wandb = False
    if args['wandb_name'] is not None:
        # Init wandb
        use_wandb = True
        init_wandb(args['wandb_name'], config)

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Declare trainer
    trainer = Trainer()

    # Define model
    quadcopter_model = QuadcopterModel(        
        n_layers=config.layers_num,
        hidden_size=config.hidden_size,
        activation_function=config.activation_function,
        learning_rate=config.learning_rate,
        model_type=config.model_type
    )
    # Move model to device
    quadcopter_model = quadcopter_model.to(device) 
    
    # Define datasets
    train_data = parse_all_datasets(config.train_data_path, exclude=config.exclude_train_data.split()) 
    val_data = parse_all_datasets(config.val_data_path)
    test_data = parse_all_datasets(config.test_data_path)

    # Plot velocity and control distribution
    graph = plot_vel_and_ctrl_distribution(train_data)
    if use_wandb:
        wandb.log({'velocities and control distribution': wandb.Image(plt)})

    # Neural network training
    trainer.fit(
        quadcopter_model,
        train_data, 
        val_data, 
        config.num_epochs, 
        config.batch_size, 
        config.rollout_size,
        config.main_metric,
        device,
        use_wandb,
        config.plot_trajectories
    )

    # Testing the neural network
    trainer.evaluate(
        quadcopter_model,
        test_data,
        device,
        plot_trajectories=config.plot_trajectories,
        use_wandb=use_wandb,
        save_to_csv=config.save_to_csv
    )

    # Save PyTorch model
    if use_wandb:
        torch.save(quadcopter_model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

if __name__ == "__main__":
    main()