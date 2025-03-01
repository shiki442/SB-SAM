# import os, sys
# root_dir = './'
# sys.path.append(root_dir)
# from config import config
# from SAM.train import evaluate_model

# dir_paths = os.path.join(root_dir, 'output')

# V_dict = {}
# V_pred_dict = {}
# V_pdc_dict = {}
# n_max = 50

# dir_paths = os.listdir(dir_paths)
# dir_path = dir_paths[-1]

# print(dir_path)

# dir_path = os.path.join(root_dir, 'output', dir_path)
# path_param = os.path.join(dir_path, 'params/config.yml')
# cfg = config.load_config(path_param)
# cfg.path.output = dir_path
# config.check_config(cfg, save_cfg=False, check_path=False)
# config.check_path_config(cfg, root_dir, create_new_file=False)

# V_pred = evaluate_model(cfg, dir_path)


# ------------------------------------------------------
from model import ncsnpp
import torch
import torch.nn as nn

if __name__ == "__main__":
    from config import config
    cfg = config.load_config()
    config.check_config(cfg)
    cfg.data.n = 125
    x = torch.randn(100, 3, 125)

    unet = ncsnpp.NCSNpp(cfg)
    print(unet)
    t = torch.rand(100)
    y = unet(x, t)
    print(x)