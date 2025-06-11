import torch.multiprocessing as mp
import os

from model.train import train_model
from model.train import evaluate_model
from config import config

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '64060'


def main(cfg):
    # for params_iter in config.generate_params():
        # config.check_config(cfg, params_iter)
    config.check_config(cfg)
    # Train the model in parallel
    if cfg.world_size <= 1:
        train_model(None, cfg)
    elif cfg.world_size >= 2:
        mp.spawn(train_model, args=(cfg,), nprocs=cfg.world_size, join=True)

    # # Evaluate the model in parallel
    # if cfg.world_size <= 1:
    #     evaluate_model(None, cfg, work_dir=cfg.path.output)
    # elif cfg.world_size >= 2:
    #     mp.spawn(evaluate_model, args=(cfg, cfg.path.output), nprocs=cfg.world_size, join=True)


if __name__ == "__main__":
    cfg = config.load_config()
    main(cfg)
