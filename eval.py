import torch.multiprocessing as mp
import os
from config import config
from model.train import evaluate_model

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '64060'

dir_paths = os.path.join('./output')

dir_paths = os.listdir(dir_paths)
dir_path = dir_paths[-1]

print(dir_path)

dir_path = os.path.join('./output', dir_path)
path_param = os.path.join(dir_path, 'params/config.yml')


def main(cfg):
    # Evaluate the model in parallel
    if cfg.world_size <= 1:
        evaluate_model(None, cfg, work_dir=cfg.path.output)
    elif cfg.world_size >= 2:
        mp.spawn(evaluate_model, args=(cfg, cfg.path.output), nprocs=cfg.world_size, join=True)


if __name__ == "__main__":
    cfg = config.load_config(path_param)
    cfg.path.output = dir_path
    cfg.data.data_eval_dir = '../SAM_dataset/data12/300K/'
    config.check_config(cfg, save_cfg=False, check_path=False)
    config.check_path_config(cfg, create_new_file=False)
    print(cfg.dynamics.temperature)
    main(cfg)
