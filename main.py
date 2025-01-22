import torch.multiprocessing as mp
import  os

from SAM.train import train_model, train_model_ddp
from config.config import cfg

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '64060'

if __name__ == "__main__":
    if cfg.world_size == 1:
        train_model(cfg)
    elif cfg.world_size >= 2:
        mp.spawn(train_model_ddp, args=cfg, nprocs=cfg.world_size)