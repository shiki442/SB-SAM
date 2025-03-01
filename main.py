import torch.multiprocessing as mp
import os

from model.train import train_model, train_model_ddp, evaluate_model
from config import config

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '64060'


def main(cfg):
    for params_iter in config.generate_params():
        config.check_config(cfg, params_iter)
        if cfg.world_size == 1:
            train_model(cfg)
        elif cfg.world_size >= 2:
            mp.spawn(train_model_ddp, args=cfg, nprocs=cfg.world_size)


if __name__ == "__main__":
    cfg = config.load_config()
    main(cfg)
