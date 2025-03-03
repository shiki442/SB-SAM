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
sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")

# ====================================   Training   ==============================================================


def train_model_ddp(rank, cfg):
    dist.init_process_group("nccl", init_method="env://",
                            rank=rank, world_size=cfg.world_size)
    torch.cuda.set_device(rank)

    # Initialize model.
    score_model = utils.create_model(cfg).to(rank)
    ddp_score = DDP(score_model, device_ids=[rank])
    optimizer = losses.get_optimizer(cfg, ddp_score.parameters())
    state = dict(optimizer=optimizer, model=ddp_score, ema=0, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(cfg.path.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(os.path.join(
        checkpoint_dir, f'checkpoint_{cfg.training.initial_step}.pth'), state, cfg.device)
    initial_step = int(state['step'])

    # Build data loader
    dataset = datasets.get_dataset(cfg)
    sampler = DistributedSampler(
        dataset, num_replicas=cfg.world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler,
                             batch_size=cfg.training.batch_size, num_workers=0)
    #  batch_size=cfg.training.batch_size, num_workers=2, persistent_workers=True)
    # Setup SDEs
    if cfg.training.sde == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=cfg.model.beta_min,
                            beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=cfg.model.beta_min,
                               beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'vesde':
        sde = sde_lib.VESDE(sigma_min=cfg.model.sigma_min,
                            sigma_max=cfg.model.sigma_max, N=cfg.model.num_scales)
        sampling_eps = 1e-8
    else:
        raise NotImplementedError(f"SDE {cfg.training.sde} unknown.")

    # Build training and evaluation functions
    reduce_mean = cfg.training.reduce_mean
    likelihood_weighting = cfg.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde, train=True, optimizer=optimizer, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)
    # Building sampling functions
    sampling_fn = sampling.get_sampling_fn(cfg, sde, sampling_eps)
    # Build evaluation function
    evaluate_fn = utils.get_evaluate_fn(cfg, dataset, save_eval=True)

    if rank == 0:
        print(f"=========================== Starting Training ===========================")
    for epoch in range(initial_step, cfg.training.n_iter + 1):
        sampler.set_epoch(epoch)
        for batch in data_loader:
            # Execute one training step
            batch = batch.to(rank).requires_grad_()
            loss = train_step_fn(state, batch)

        # Print the averaged training loss so far.
        if epoch % cfg.log.print_interval == 0 and rank == 0:
            print(f'epoch: {epoch}\t loss: {loss.item():.5f}')
        if (epoch != 0 and epoch % cfg.log.save_interval == 0) or epoch == cfg.training.n_iter:
            if rank == 0:
                save_checkpoint(os.path.join(
                    checkpoint_dir, f'checkpoint_{epoch}.pth'), state)
    if rank == 0:
        print(
            f"=========================== Finished Training ===========================\n")

    if rank == 0:
        # Generate samples
        x_pred = sampling_fn(score_model)

        # Evaluate the model
        V_pred = evaluate_fn(x_pred)
        return V_pred

    dist.destroy_process_group()
    return


def train_model(cfg):
    # Initialize model.
    score_model = utils.create_model(cfg)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=0, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(cfg.path.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(os.path.join(
        checkpoint_dir, f'checkpoint_{cfg.training.initial_step}.pth'), state, cfg.device)
    initial_step = int(state['step'])

    # Build data loader
    dataset = datasets.get_dataset(cfg)
    data_loader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Setup SDEs
    if cfg.training.sde == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=cfg.model.beta_min,
                            beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=cfg.model.beta_min,
                               beta_max=cfg.model.beta_max, N=cfg.model.num_scales)
        sampling_eps = 1e-3
    elif cfg.training.sde == 'vesde':
        sde = sde_lib.VESDE(sigma_min=cfg.model.sigma_min,
                            sigma_max=cfg.model.sigma_max, N=cfg.model.num_scales)
        sampling_eps = 1e-8
    else:
        raise NotImplementedError(f"SDE {cfg.training.sde} unknown.")

    # Build training and evaluation functions
    reduce_mean = cfg.training.reduce_mean
    likelihood_weighting = cfg.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde, train=True, optimizer=optimizer, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)
    # Building sampling functions
    sampling_fn = sampling.get_sampling_fn(cfg, sde, sampling_eps)
    # Build evaluation function
    evaluate_fn = utils.get_evaluate_fn(cfg, dataset, save_eval=True)

    print(f"=========================== Starting Training ===========================")
    tqdm_epoch = tqdm(range(initial_step, cfg.training.n_iter + 1),
                      desc="Training: Optimizer=Adam")
    for epoch in tqdm_epoch:
        for batch in data_loader:
            # Execute one training step
            batch = batch.to(cfg.device).requires_grad_()
            # batch = batch.requires_grad_()
            loss = train_step_fn(state, batch)

        # Print the averaged training loss so far.
        if epoch % cfg.log.print_interval == 0:
            tqdm.write(f'epoch: {epoch}\t loss: {loss.item():.5f}')
        if (epoch != 0 and epoch % cfg.log.save_interval == 0) or epoch == cfg.training.n_iter:
            save_checkpoint(os.path.join(
                checkpoint_dir, f'checkpoint_{epoch}.pth'), state)
    print(f"=========================== Finished Training ===========================\n")

    # Generate samples
    x_pred = sampling_fn(score_model)

    # Evaluate the model
    V_pred = evaluate_fn(x_pred)
    return V_pred


def evaluate_model(cfg, work_dir):
    # Initialize model.
    score_model = utils.create_model(cfg)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=0, step=0)

    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(os.path.join(
        work_dir, f'checkpoints/checkpoint_{cfg.training.n_iter}.pth'), state, cfg.device)
    score_model = state['model']

    dataset = datasets.get_dataset(cfg)

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
    evaluate_fn = utils.get_evaluate_fn(cfg, dataset)

    # Generate samples
    x_pred = sampling_fn(score_model)
    # Evaluate the model
    V_pred = evaluate_fn(x_pred)
    return V_pred


def load_eval(cfg, dir_path):
    eval_path = os.path.join(dir_path, 'eval/eval.pth')
    eval = torch.load(eval_path)
    V_pred = eval['V_pred']
    return V_pred

    # score_fn = utils.get_score_fn(sde, score_model, train=False)
    # x_pred = sampling.Euler_Maruyama_sampler(score_fn, shape, init_x, diffusion, batch_size=cfg.sampler.ntrajs, n_steps=cfg.model.num_scales, eps=cfg.eps)

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
