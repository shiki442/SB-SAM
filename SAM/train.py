import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.optim import Adam, LBFGS

import sys
sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")
# from SAM.nn import marginal_prob_std
from SAM.sde_lib import VESDE
from SAM.datasets import get_dataloader
from SAM import cfg, datasets, utils, losses, sde_lib
from tqdm import tqdm

###====================================   Training   ==============================================================
def train_model_ddp(rank, world_size, score_model, dataset, lr=1e-3, batch_size=1000, n_epochs=5000, print_interval=100):
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    local_model = score_model.to(rank)
    ddp_score = DDP(local_model, device_ids=[rank])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    optimizer = Adam(ddp_score.parameters(), lr=lr)
    train_loss = []
    
    marginal_prob_std = lambda t : VESDE.marginal_prob_std(cfg.sigma_min, cfg.sigma_max, t)

    if rank == 0:
        print(f"=========================== Starting Training ===========================")
    
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        avg_loss = 0.
        for x in data_loader:
            x = x.to(rank).requires_grad_()
            loss = losses.loss_dsm(ddp_score, x, marginal_prob_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Print the averaged training loss.
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        avg_loss = loss.item() / world_size
        train_loss.append(avg_loss)
        if epoch%print_interval==0 and rank==0:
            print(f'epoch: {epoch}\t loss: {avg_loss:.5f}')
    if rank == 0:
        print(f"=========================== Finished Training ===========================\n")
    dist.destroy_process_group()
    return train_loss


def train_model(cfg):
    # Initialize model.
    score_model = utils.create_model(cfg)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())

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
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {cfg.training.sde} unknown.")

    train_step_fn = losses.get_step_fn(sde, train=True, optimizer=optimizer)

    train_loss = []
    
    print(f"=========================== Starting Training ===========================")
    tqdm_epoch = tqdm(range(cfg.n_epochs), desc="Training: Optimizer=Adam")
    for epoch in tqdm_epoch:
        for batch in data_loader:
            batch = batch.requires_grad_()
            loss = train_step_fn(score_model, batch)

        # Print the averaged training loss so far.
        train_loss.append(loss.item())
        if epoch%cfg.print_interval==0:
            tqdm.write(f'epoch: {epoch}\t loss: {loss.item():.5f}')

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
    print(f"=========================== Finished Training ===========================\n")
    return train_loss

