import torch
import time
from SAM import cfg, datasets
from tqdm import tqdm
import math

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