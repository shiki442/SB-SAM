import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset

import numpy as np

import sys, os, yaml
sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")
# from SAM.nn import marginal_prob_std
from SAM.sde_lib import VESDE
from SAM.datasets import get_dataloader
from SAM import datasets, utils, losses, sde_lib, sampling, nn
from SAM.checkpoint import save_checkpoint, restore_checkpoint
from tqdm import tqdm

###====================================   Training   ==============================================================
def train_model_ddp(rank, cfg):
    # dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)

    # local_model = score_model.to(rank)
    # ddp_score = DDP(local_model, device_ids=[rank])
    
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    # optimizer = Adam(ddp_score.parameters(), lr=lr)
    # train_loss = []
    
    # marginal_prob_std = lambda t : VESDE.marginal_prob_std(cfg.sigma_min, cfg.sigma_max, t)

    # if rank == 0:
    #     print(f"=========================== Starting Training ===========================")
    
    # for epoch in range(n_epochs):
    #     sampler.set_epoch(epoch)
    #     avg_loss = 0.
    #     for x in data_loader:
    #         x = x.to(rank).requires_grad_()
    #         loss = losses.loss_dsm(ddp_score, x, marginal_prob_std)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #     # Print the averaged training loss.
    #     dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    #     avg_loss = loss.item() / world_size
    #     train_loss.append(avg_loss)
    #     if epoch%print_interval==0 and rank==0:
    #         print(f'epoch: {epoch}\t loss: {avg_loss:.5f}')
    # if rank == 0:
    #     print(f"=========================== Finished Training ===========================\n")
    # dist.destroy_process_group()
    # return train_loss
    return


def train_model(cfg):
    # Initialize model.
    score_model = utils.create_model(cfg)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=0, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(cfg.path_output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(os.path.join(checkpoint_dir, "checkpoint.pth"), state, cfg.device)
    initial_step = int(state['step'])

    # Build data loader
    data_loader = datasets.get_dataloader(cfg)

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
    train_step_fn = losses.get_step_fn(sde, train=True, optimizer=optimizer)
    

    
    print(f"=========================== Starting Training ===========================")
    tqdm_epoch = tqdm(range(initial_step, cfg.n_epochs), desc="Training: Optimizer=Adam")
    for epoch in tqdm_epoch:
        for batch in data_loader:
            # Execute one training step
            batch = batch.requires_grad_()
            loss = train_step_fn(state, batch)

        # Print the averaged training loss so far.
        if epoch%cfg.print_interval==0:
            tqdm.write(f'epoch: {epoch}\t loss: {loss.item():.5f}')
        if epoch%cfg.save_interval==0:
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'), state)

    print(f"=========================== Finished Training ===========================\n")

    # Building sampling functions
    sampling_fn = sampling.get_sampling_fn(cfg, sde, sampling_eps)

    x_pred = sampling_fn(score_model)

    return loss

    # score_fn = utils.get_score_fn(sde, score_model, train=False)
    # x_pred = sampling.Euler_Maruyama_sampler(score_fn, shape, init_x, diffusion, batch_size=cfg.ntrajs_sample, n_steps=cfg.n_steps, eps=cfg.eps)

    # tqdm_epoch = tqdm(range(50), desc="Training: Optimizer=LBFGS")
    # for epoch in tqdm_epoch:
    #     for x in data_loader:
            
    #         def closure():
    #             # x = x.requires_grad_()
    #             optimizer_lbfgs.zero_grad()
    #             loss = loss_dsm(score_model, x, marginal_prob_std) 
    #             loss.backward()
    #             return loss
            
    #         optimizer_lbfgs.step(closure)
    #     # Print the averaged training loss so far.
    #     train_loss.append(loss.item())
    #     if epoch%5==0:
    #         tqdm.write(f'epoch: {epoch}\t loss: {loss.item():.5f}')

    # optimizer = optimizer_lbfgs
    # loss_value = closure().item()
    # optimizer.step(closure)