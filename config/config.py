from ml_collections import ConfigDict
from datetime import datetime
import torch
import argparse, os, itertools
import yaml
import math

def load_config(config_file=None):
    if config_file is None:
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
    if data.grid_step == 0:
        data.grid_step = (data.max_grid-data.min_grid) / (data.n_all_per_dim-1)
        n_out = math.ceil((data.max_grid-data.max_sam)/data.grid_step) + math.ceil((data.min_sam-data.min_grid)/data.grid_step)
        data.n_sam_per_dim = data.n_all_per_dim - n_out
    elif data.cfg_flag == 1:
        data.max_grid = (data.n_all_per_dim-1) * data.grid_step
        data.min_grid = 0.
        data.max_sam = (data.n_sam_per_dim-1) * data.grid_step
        data.min_sam = 0.
    data.n_all = data.n_all_per_dim ** data.d
    data.n_sam = data.n_sam_per_dim ** data.d
    data.nd = data.d * data.n_sam

def check_path_config(cfg):
    path = cfg.path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path.output = f'./output/{timestamp}_N{cfg.data.n_all}_n{cfg.data.n_sam}_k{cfg.model.k_near}_d{cfg.data.d}/'
    path.checkpoints = os.path.join(path.output, "checkpoints")
    path.eval = os.path.join(path.output, "eval")
    path.params = os.path.join(path.output, "params", "config.yml")
    os.makedirs(path.output, exist_ok=True)
    os.makedirs(path.checkpoints, exist_ok=True)
    os.makedirs(path.eval, exist_ok=True)
    os.makedirs(os.path.dirname(path.params), exist_ok=True)

def save_config(cfg):
    cfg.device = None
    cfg.world_size = None
    with open(cfg.path.params, 'w') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)

def check_device_config(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.world_size = torch.cuda.device_count()

def generate_params():
    k_nearest_list = [5]
    n_list = [5]

    # k_nearest_list = [5,10]
    # n_list = [5,10,15,20,25]

    # d = 3
    # n_list = [4,8,12,16,20]
    # k_nearest = [5,10,15,20]

    for k, n in itertools.product(k_nearest_list, n_list):
        yield {'k_nearest': k, 'n': n}

def check_config(cfg, params=None, save_cfg=True, check_path=True):
    if params is not None:
        cfg.model.k_near = params['k_nearest']
        cfg.data.n_sam_per_dim = params['n']
    check_data_config(cfg)
    if check_path:
        check_path_config(cfg)
    if save_cfg:
        save_config(cfg)
    check_device_config(cfg)
