import argparse
import os
import sys
import warnings
import json
import tqdm 
import numpy as np
import pandas as pd
from dotmap import DotMap
from typing import List
import multiprocessing 

from arguments import Parser
from src.common.data_utils import *
from src.dataloaders import init_dataloader
from src.models import init_model
from src.trainers import init_trainer
from datetime import datetime
import debugpy

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()
debugpy.breakpoint()
print('break on this line')

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def main(sys_argv: List[str] = None):
    # Parser
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    configs = Parser(sys_argv).parse()
    args = DotMap(configs, _dynamic=False)

    print(args)
    # Dataset
    dataset_path = args.local_data_folder + '/' + args.dataset_type

    match_df = {}
    match_df['train'] = pd.read_csv(dataset_path + '/train.csv', index_col=0)
    match_df['val'] =  pd.read_csv(dataset_path + '/val.csv', index_col=0)
    match_df['test'] =  pd.read_csv(dataset_path + '/test.csv', index_col=0)
    user_history_array = np.load(dataset_path + '/user_history.npy', mmap_mode='r+')

    # Print shapes of each dataset
    print("Train dataset shape:", match_df['train'].shape)
    print("Validation dataset shape:", match_df['val'].shape)
    print("Test dataset shape:", match_df['test'].shape)
    print("shape of user_history_array", user_history_array.shape)

    # Load data in memory if your memory (>30Gb)
    # user_history_array = read_large_array(dataset_path + '/user_history.npy')

    with open(dataset_path + '/categorical_ids.json', 'r') as f:
        categorical_ids = json.load(f)
    with open(dataset_path + '/feature_to_array_idx.json', 'r') as f:
        feature_to_array_idx = json.load(f)

    args.num_champions = len(categorical_ids['champion'])
    args.num_roles = len(categorical_ids['role'])
    args.num_teams = len(categorical_ids['team'])
    args.num_outcomes = len(categorical_ids['win'])
    args.num_stats = len(categorical_ids['stats'])
    # DataLoader
    train_dataloader, val_dataloader, test_dataloader = init_dataloader(args, 
                                                                        match_df, 
                                                                        user_history_array, 
                                                                        feature_to_array_idx)

    print(args)

    # Model
    model = init_model(args)

    # Trainer
    trainer = init_trainer(args, 
                           train_dataloader, 
                           val_dataloader, 
                           test_dataloader, 
                           model)
    trainer.train()

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()