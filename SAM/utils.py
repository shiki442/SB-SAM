# Practical tools used by main.py

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import sys, time
import math
import random

sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")

from SAM import cfg

###====================================   Set Seed   ==============================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SamDataset(Dataset):
    def __init__(self, x_train):
        super().__init__()
        self.x_train = x_train
        self.shape = [x_train.shape[0], x_train.shape[1]*x_train.shape[2]]

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.flatten(self.x_train[idx], start_dim=-2)


class MyDataset(Dataset):
    def __init__(self, x_train):
        super().__init__()
        self.x_train = x_train
        self.shape = x_train.shape

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx]



###====================================   Sampling   ==============================================================
def Euler_Maruyama_sampler(score_model,
                           shape,
                           init_x,
                           diffusion_coeff, 
                           batch_size=64, 
                           n_steps=1000, 
                           eps=1e-3):
    start_time = time.time()

    t = torch.ones(batch_size, device=cfg.device)
    shape[0] = batch_size
    # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None]
    # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None, None] # Unet
    time_steps = torch.linspace(1., eps, n_steps, device=cfg.device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    print(f"=========================== Start Sampling ===========================")
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="Sampling"):
            batch_time_step = torch.ones(batch_size, device=cfg.device) * time_step
            g = diffusion_coeff(batch_time_step)[:, None]
            # g = diffusion_coeff(batch_time_step)[:, None, None] # Unet
            mean_x = x + (g**2) * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)    

    end_time = time.time()
    print(f"Num of trajectories: {batch_size}")
    print(f"Num of SDE Steps: {n_steps}")
    print(f"Smallest time step: {eps}")
    print(f"Total time = {(end_time-start_time)/60.:.5f}m")
    print(f"=========================== Finished Sampling ===========================\n")
    return mean_x


def pc_sampler(score_model,
                shape,
                init_x,
                diffusion_coeff, 
                batch_size=64, 
                n_steps=1000,
                snr=0.01,
                eps=1e-3):
    start_time = time.time()

    t = torch.ones(batch_size, device=cfg.device)
    shape[0] = batch_size
    # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None]
    # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None, None] # Unet
    time_steps = torch.linspace(1., eps, n_steps, device=cfg.device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    print(f"=========================== Start Sampling ===========================")
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="Sampling"):
            batch_time_step = torch.ones(batch_size, device=cfg.device) * time_step

            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad, dim=-1).mean()
            # grad_norm = torch.tensor(25.0)
            noise_norm = math.sqrt(x.shape[1])
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)[:, None]
            # g = diffusion_coeff(batch_time_step)[:, None, None] # Unet
            mean_x = x + (g**2) * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)    

    end_time = time.time()
    print(f"Num of trajectories: {batch_size}")
    print(f"Num of SDE Steps: {n_steps}")
    print(f"Smallest time step: {eps}")
    print(f"Total time = {(end_time-start_time)/60.:.5f}m")
    print(f"=========================== Finished Sampling ===========================\n")
    return mean_x




###====================================   Stress   ==============================================================
def potential(x_all, mean, d2V):
    u = x_all - mean
    V = 0.
    for i in range(1):
        f = u[:,:,i] @ d2V[i]
        V += 0.5 * torch.mean(f * u[:,:,i])
        # V += torch.mean(torch.pow(u, 4))
    return V

def force(x_all, mean, d2V):
    f = torch.zeros_like(x_all)
    for i in range(cfg.d):
        f[:,:,i] = 2*(x_all[:,:,i] - mean[:, i]) @ d2V[i]
    return f

def stress_LMC(sc, x, x_grid):
    x = x.to(cfg.device)
    x_flatten = torch.flatten(x, start_dim=-2)
    x_grid = x_grid.to(cfg.device)
    t = torch.ones(x_flatten.shape[0], device=cfg.device) * 0.01
    f_flatten = sc(x_flatten, t)
    f = f_flatten.view(x.shape)
    # p = -torch.sum(f * (x-x_grid), axis=2)
    return stress_MD(f, x_grid)

def stress_MD(f, x_grid_sam):
    ft = f.transpose(-1, -2)
    p = ft @ x_grid_sam
    return torch.mean(p, axis=0), f
