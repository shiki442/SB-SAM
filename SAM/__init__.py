from ml_collections import ConfigDict
from datetime import datetime
import torch
import argparse, sys, shutil, math, os
import yaml

parser = argparse.ArgumentParser(description="Basic paser")
parser.add_argument("--config_path", type=str,
                    help="Path to the configuration file",
                    default="")
args = parser.parse_args()
config_file = args.config_path
# config_file = './params.yml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

cfg = ConfigDict(config)

SEED = 1213

cfg.n_all = cfg.n_all_per_dim ** cfg.d
cfg.grid_step = (cfg.max_grid-cfg.min_grid) / (cfg.n_all_per_dim-1)

n_out = math.ceil((cfg.max_grid-cfg.max_sam)/cfg.grid_step) + math.ceil((cfg.min_sam-cfg.min_grid)/cfg.grid_step)
cfg.n_sam_per_dim = cfg.n_all_per_dim - n_out
cfg.n_sam = cfg.n_sam_per_dim ** cfg.d
cfg.nd = cfg.d * cfg.n_sam

cfg.k0 = cfg.k0 * (6.0/cfg.grid_step)**2

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

cfg.path_output = f'./output/{timestamp}_N{cfg.n_all}_n{cfg.n_sam}_d{cfg.d}/'
cfg.path_checkpoints = os.path.join(cfg.path_output, "checkpoints")
cfg.path_params = os.path.join(cfg.path_output, "params")

os.makedirs(cfg.path_params, exist_ok=True)
with open('config.yml', 'w') as f:
    yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.world_size = torch.cuda.device_count()