from ml_collections import ConfigDict
from datetime import datetime
import torch
import argparse
import os
import itertools
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
    if data.crystal == 'SC':
        data.max_grid = (data.nx_max-1) * data.grid_step
        data.min_grid = 0.
        data.max_sam = (data.nx-1) * data.grid_step
        data.min_sam = 0.
        assert data.nx <= data.nx_max
        data.n_max = data.nx_max ** data.d
        data.n = data.nx ** data.d
        data.nf = data.d * data.n
    if data.crystal == 'BCC':
        data_dir = os.path.join(data.data_dir, 'data_params.txt')
        params = read_params(data_dir)
        data.d = params['ndim']
        data.grid_step = params['a0']
        data.nx_max = params['nx']
        data.n_max = params['num_atoms']
        assert data.nx <= data.nx_max
        assert cfg.training.ntrajs <= params['nsample']
        assert cfg.sampler.ntrajs <= params['nsample']
        data.na = params['na']
        data.n = data.nx ** data.d * data.na
        data.nf = params['nf']
    assert cfg.training.batch_size <= cfg.training.ntrajs


def read_params(filename):
    params = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ',' in line:
                key, value = line.split(',')
                key = key.strip()
                value = value.strip()
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = []
            else:
                if 'defm' in params:
                    params['defm'].append([float(x) for x in line.split()])
                else:
                    params['defm'] = [[float(x) for x in line.split()]]
    return params


def check_path_config(cfg, word_dir='./', create_new_file=True):
    path = cfg.path
    if path.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.output = os.path.join(
            word_dir, f'output/{timestamp}_N{cfg.data.n_max}_n{cfg.data.n}_k{cfg.dynamics.k_near}_d{cfg.data.d}/')
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
        yaml.dump(cfg.to_dict(), f, default_flow_style=False,
                  allow_unicode=True)


def check_device_config(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.world_size = torch.cuda.device_count()


def generate_params():
    # d = 1
    # k_nearest_list = [5]
    # n_list = [3]

    # k_nearest_list = [5, 10, 15, 20]
    # n_list = [6]

    # d = 3
    n_list = [4,8,12,16,20]
    k_nearest_list = [5,10,15,20]

    for k, n in itertools.product(k_nearest_list, n_list):
        yield {'k_nearest': k, 'n': n}


def check_config(cfg, params_iter=None, save_cfg=True, check_path=True):
    if params_iter is not None:
        cfg.dynamics.k_near = params_iter['k_nearest']
        cfg.data.nx = params_iter['n']
    check_data_config(cfg)
    if check_path:
        check_path_config(cfg)
    if save_cfg:
        save_config(cfg)
    check_device_config(cfg)
