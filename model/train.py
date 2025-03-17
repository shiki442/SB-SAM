from tqdm import tqdm
from config.config import load_config
from model.checkpoint import save_checkpoint, restore_checkpoint
from model import datasets, utils, losses, sde_lib, sampling, nn, ncsnpp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset

import sys
import os
import logging

sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")

# ==========================   Training   ==========================


def train_model(rank, cfg):
    if rank is not None:
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=cfg.world_size)
        torch.cuda.set_device(rank)
        utils.set_seed(rank)
    else:
        utils.set_seed()

    # Initialize model.
    score_model = utils.create_model(cfg)
    if rank is not None:
        score_model = score_model.to(rank)
        ddp_score = DDP(score_model, device_ids=[rank])
        model = ddp_score
    else:
        model = score_model.to(cfg.device)
    
    optimizer = losses.get_optimizer(cfg, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=0, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(cfg.path.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(os.path.join(
        checkpoint_dir, f'checkpoint_{cfg.training.initial_step}.pth'), state, cfg.device)
    initial_step = int(state['step'])

    # Build data loader
    dataset = datasets.get_dataset(cfg, mode='train')
    if rank is not None:
        sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=rank, shuffle=True)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.training.batch_size, num_workers=0)
    else:
        data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Setup SDEs
    if cfg.training.sde == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=cfg.model.beta_min, beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=cfg.model.beta_min, beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'vesde':
        sde = sde_lib.VESDE(sigma_min=cfg.model.sigma_min, sigma_max=cfg.model.sigma_max, N=cfg.model.num_scales)
        sampling_eps = 1e-8
    else:
        raise NotImplementedError(f"SDE {cfg.training.sde} unknown.")

    # Build training and evaluation functions
    reduce_mean = cfg.training.reduce_mean
    likelihood_weighting = cfg.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimizer=optimizer, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)
    sampling_fn = sampling.get_sampling_fn(cfg, sde, sampling_eps)
    evaluate_fn = utils.get_evaluate_fn(cfg, save_eval=True)

    if rank is None or rank == 0:
        tqdm.write(f"========================== Starting Training ==========================")
    # epochs_range = range(initial_step, cfg.training.n_iter)
    # tqdm_bar = tqdm(epochs_range, desc="Training", mininterval=2, ncols=0)    
    for epoch in range(initial_step, cfg.training.n_iter):
        if rank is not None:
            sampler.set_epoch(epoch)
        for batch, tau in data_loader:
            batch = batch.to(rank if rank is not None else cfg.device)
            tau = tau.to(rank if rank is not None else cfg.device)
            loss = train_step_fn(state, batch, tau)

        if rank is None or rank == 0:
            if (epoch+1) % cfg.log.print_interval == 0:
                tqdm.write(f'epoch: {epoch+1}\t loss: {loss.item():.5f}')
            if (epoch+1) % cfg.log.save_interval == 0 or (epoch+1) == cfg.training.n_iter:
                save_checkpoint(checkpoint_dir, state, epoch+1)
    torch.cuda.empty_cache()

    if rank is None or rank == 0:
        tqdm.write(f"========================== Finished Training ==========================\n")

    if rank is not None:
        dist.destroy_process_group()
    return None


def evaluate_model(rank, cfg, work_dir):
    if rank is not None:
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=cfg.world_size)
        torch.cuda.set_device(rank)
        utils.set_seed(rank)
    else:
        utils.set_seed()

    # Initialize model.
    score_model = utils.create_model(cfg)
    if rank is not None:
        score_model = score_model.to(rank)
        ddp_score = DDP(score_model, device_ids=[rank])
        model = ddp_score
    else:
        model = score_model.to(cfg.device)
    
    optimizer = losses.get_optimizer(cfg, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=0, step=0)

    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(os.path.join(
        work_dir, f'checkpoints/checkpoint_{cfg.training.n_iter}.pth'), state, cfg.device)
    score_model = state['model']

    # Setup SDEs
    if cfg.training.sde == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=cfg.model.beta_min,
                            beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-5
    elif cfg.training.sde == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=cfg.model.beta_min,
                               beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'vesde':
        sde = sde_lib.VESDE(sigma_min=cfg.model.sigma_min,
                            sigma_max=cfg.model.sigma_max, N=cfg.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {cfg.training.sde} unknown.")

    # Building sampling functions
    sampling_fn = sampling.get_sampling_fn(cfg, sde, sampling_eps)
    # Build evaluation function
    evaluate_fn = utils.get_evaluate_fn(cfg, save_eval=True)

    # Generate samples
    x_pred = sampling_fn(score_model)

    if rank is not None:
        x_pred_list = [torch.zeros_like(x_pred) for _ in range(cfg.world_size)]
        dist.all_gather(x_pred_list, x_pred)
        x_pred = torch.cat(x_pred_list, dim=0)

    if rank is None or rank == 0:
        V_pred = evaluate_fn(x_pred)
        return V_pred

    if rank is not None:
        dist.destroy_process_group()
    return None


def load_eval(cfg, dir_path):
    eval_path = os.path.join(dir_path, 'eval/eval.pth')
    eval = torch.load(eval_path)
    V_pred = eval['V_pred']
    return V_pred
