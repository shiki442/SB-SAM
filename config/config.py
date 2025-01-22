from ml_collections import ConfigDict
from datetime import datetime
import torch
import argparse, os
import yaml
import math

def load_config():
    parser = argparse.ArgumentParser(description="Basic paser")
    parser.add_argument("--config_path", type=str,
                        help="Path to the configuration file",
                        default="")
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    cfg = ConfigDict(config)
    return cfg

def check_data_config(cfg):
    data = cfg.data
    if data.grid_step is None:
        data.grid_step = (data.max_grid-data.min_grid) / (data.n_all_per_dim-1)
        n_out = math.ceil((data.max_grid-data.max_sam)/data.grid_step) + math.ceil((data.min_sam-data.min_grid)/data.grid_step)
        data.n_sam_per_dim = data.n_all_per_dim - n_out
    elif data.min_grid is None:
        data.max_grid = 0.5*(data.n_all_per_dim-1) * data.grid_step
        data.min_grid = -0.5*(data.n_all_per_dim-1) * data.grid_step
        data.max_sam = 0.5*(data.n_sam_per_dim-1) * data.grid_step
        data.min_sam = -0.5*(data.n_sam_per_dim-1) * data.grid_step
    data.n_all = data.n_all_per_dim ** data.d
    data.n_sam = data.n_sam_per_dim ** data.d
    data.nd = data.d * data.n_sam

def check_path_config(cfg):
    path = cfg.path
    if path.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.output = f'./output/{timestamp}_N{cfg.data.n_all}_n{cfg.data.n_sam}_d{cfg.data.d}/'
    path.checkpoints = os.path.join(path.output, "checkpoints")
    path.eval = os.path.join(path.output, "eval")
    path.params = os.path.join(path.output, "params", "config.yml")
    os.makedirs(path.output, exist_ok=True)
    os.makedirs(path.checkpoints, exist_ok=True)
    os.makedirs(path.eval, exist_ok=True)
    os.makedirs(os.path.dirname(path.params), exist_ok=True)

def save_config(cfg):
    with open(cfg.path.params, 'w') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)

def check_device_config(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.world_size = torch.cuda.device_count()

cfg = load_config()
check_data_config(cfg)
check_path_config(cfg)
save_config(cfg)
check_device_config(cfg)