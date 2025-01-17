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
from SAM.utils import SamDataset
from SAM import cfg
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
            loss = loss_dsm(ddp_score, x, marginal_prob_std)
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


def train_model(score_model, dataset, lr=1e-3, batch_size=1000, n_epochs=5000, print_interval=100, device='cuda'):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(score_model.parameters(), lr=lr, weight_decay=0)
    optimizer_lbfgs = LBFGS(score_model.parameters(), lr=0.1, max_iter=10, max_eval=None, tolerance_grad=1e-09, tolerance_change=1e-11)

    marginal_prob_std = lambda t : VESDE.marginal_prob_std(cfg.sigma_min, cfg.sigma_max, t)

    train_loss = []
    
    print(f"=========================== Starting Training ===========================")
    tqdm_epoch = tqdm(range(n_epochs), desc="Training: Optimizer=Adam")
    for epoch in tqdm_epoch:
        for x in data_loader:
            x = x.requires_grad_()
            loss = loss_dsm(score_model, x, marginal_prob_std) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Print the averaged training loss so far.
        train_loss.append(loss.item())
        if epoch%print_interval==0:
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



###====================================   Loss Function   ==============================================================
def DSM_loss(model, x, sigma=1.0e-1):
    """The loss function for training score-based generative models.
    Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    sigma: Noise level for DSM.
    """
    perturbed_x = x + torch.randn_like(x) * sigma
    z = - 1 / (sigma ** 2) * (perturbed_x - x)
    scores = model(perturbed_x)
    z = z.view(z.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - z) ** 2).sum(dim=-1).mean(dim=0)
    return loss


def loss_dsm(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
        time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    # perturbed_x = x + z * std[:, None, None] # Unet
    perturbed_x = x + z * std[:, None]
    score = model(perturbed_x, random_t)
    # loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=1)) # Unet
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
    return loss


def loss_ssm(model, samples, sigma=0.1):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    score = model(perturbed_samples)
    div_score = model.div(perturbed_samples)
    loss = torch.sum(score**2) + 2 * div_score
    return torch.mean(loss)