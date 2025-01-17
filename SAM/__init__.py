from ml_collections import ConfigDict
from datetime import datetime
import torch
import argparse, sys, shutil, math
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SEED = 1213

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg.n_all = cfg.n_all_per_dim ** cfg.d
cfg.grid_step = (cfg.max_grid-cfg.min_grid) / (cfg.n_all_per_dim-1)

n_out = math.ceil((cfg.max_grid-cfg.max_sam)/cfg.grid_step) + math.ceil((cfg.min_sam-cfg.min_grid)/cfg.grid_step)
cfg.n_sam_per_dim = cfg.n_all_per_dim - n_out
cfg.n_sam = cfg.n_sam_per_dim ** cfg.d
cfg.nd = cfg.d * cfg.n_sam

cfg.k0 = cfg.k0 * (6.0/cfg.grid_step)**2

cfg.path_output = './output/'
cfg.path_model_output = f"{cfg.path_output}models/sc_{timestamp}_N{cfg.n_all}_n{cfg.n_sam}_d{cfg.d}.pth"
cfg.path_figure_output = f"{cfg.path_output}figures/fig_{timestamp}_N{cfg.n_all}_n{cfg.n_sam}_d{cfg.d}/"
cfg.path_params = config_file
cfg.path_params_output = f"{cfg.path_output}params/params_{timestamp}_N{cfg.n_all}_n{cfg.n_sam}_d{cfg.d}.yml"
