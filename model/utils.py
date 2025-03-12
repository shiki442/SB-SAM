# Practical tools used by main.py

import torch
import sys
import os
import math

from model.datasets import process_data
import matplotlib.pyplot as plt

sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    return score_model


def get_score_fn(sde, model, train):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function."""

    def score_fn(x, t, tau):
        score = model(x, t)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = -score / std[:, None, None]
        return score / tau[:, None, None]

    return score_fn

# ==========================   evaluate   ==========================


def get_evaluate_fn(cfg, dataset, save_eval=True):
    mean_true = dataset.mean
    std_true = dataset.std
    x_true = dataset.x_sam
    potential_fn = get_potential_fn(cfg)

    def evaluate_fn(x_pred):
        x_pred = process_data(x_pred, mean_true, cfg)

        mean_pred = torch.mean(x_pred, axis=0)
        std_pred = torch.std(x_pred, axis=0)
        err_mean = torch.mean(
            torch.abs((mean_pred - mean_true)/cfg.data.grid_step))
        err_std = torch.mean(torch.abs((std_pred - std_true)/std_true))
        # print(f"Mean_pred: {mean_pred:.5f}\n, mean_true: {mean_true:.5f}")
        # print(f"Std_pred: {mean_pred:.5f}\n, std_true: {mean_true:.5f}")
        print(f"Mean Error: {err_mean:.5f}, Std Error: {err_std:.5f}")

        V_true = potential_fn(x_true, mean_true)
        V_pred = potential_fn(x_pred, mean_pred)
        err_V = torch.abs((V_pred - V_true)/V_true)
        print(f"Predicted Potential: {V_pred:.5f}")
        print(f"Potential Error: {err_V:.5f}")

        if save_eval:
            eval = dict(x_pred=x_pred, V_pred=V_pred, V_true=V_true)
            torch.save(eval, os.path.join(cfg.path.eval, 'eval.pth'))

            create_and_save_hist(
                x_pred[:, 0, 1], x_true[:, 0, 1], cfg.path.eval)
            create_and_save_rdf(x_pred, x_true, cfg.path.eval, cfg.data.grid_step)
        return V_pred

    return evaluate_fn


def get_pdc_evaluate_fn(cfg, dataset):
    mean_true = dataset.mean
    potential_fn = get_potential_fn(cfg)

    def evaluate_fn(x_pdc):
        V_pdc = potential_fn(x_pdc, mean_true)
        return V_pdc

    return evaluate_fn

# ===================================   Stress   ===================================


def get_potential_fn(cfg):

    def potential(x, mean):
        n = x.shape[2]
        Fmat = get_force_mat(cfg, n)
        u = x - mean
        V = 0.
        for i in range(cfg.data.d):
            f = u[:, i, :] @ Fmat[i]
            V += 0.5 * torch.mean(f * u[:, i, :])
        return V

    return potential


def get_force_mat(cfg, n):
    Fmat = torch.zeros((cfg.data.d, n, n))
    diag_vals = [10., -9., 8., -7., 6., -5.]
    # 构造三对角矩阵
    for i in range(len(diag_vals)):
        if i <= n:
            diag = diag_vals[i] * torch.ones(n - i)
            Fmat[0] = Fmat[0] + \
                torch.diag(diag, diagonal=i) + torch.diag(diag, diagonal=-i)
    return Fmat.to(cfg.device)


def create_and_save_hist(x_pred, x_true, path):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    axes[0].hist(x_pred.cpu().numpy(), bins=50,
                 edgecolor='black', density=True)
    axes[0].set_title(f'pred data: x')
    axes[1].hist(x_true.cpu().numpy(), bins=50,
                 edgecolor='black', density=True)
    axes[1].set_title(f'true data: x')
    path = os.path.join(path, 'hist.png')
    plt.savefig(path)
    plt.close()


def compute_rdf(positions, box_size, r_max, r_min, bin_width):
    num_bins = int(r_max / bin_width)
    num_bins = num_bins + 1 if math.modf(r_max / bin_width)[0] > 0.99 else num_bins
    num_bins_cutoff = int(r_min / bin_width)
    rdf_hist = torch.zeros(num_bins, device=positions.device)
    num_particles = positions.shape[0] * positions.shape[-1]
    positions = positions - torch.floor(positions / box_size) * box_size
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                positions_shifted = positions.clone()
                positions_shifted[:, 0] *= i
                positions_shifted[:, 1] *= j
                positions_shifted[:, 2] *= k
                r = torch.linalg.norm(positions_shifted, axis=1)
                mask = r < r_max
                bin_index = (r[mask] / bin_width).to(torch.int64)
                rdf_hist += torch.bincount(bin_index)

    r_values = (torch.arange(num_bins, device=positions.device) + 0.5) * bin_width
    shell_volumes = (4 / 3) * torch.pi * \
        ((r_values + bin_width) ** 3 - r_values ** 3)
    ideal_density = num_particles / (box_size ** 3)
    rdf_hist = rdf_hist / (shell_volumes * ideal_density)
    rdf_hist[0:num_bins_cutoff] = 0

    return r_values, rdf_hist


def create_and_save_rdf(x_pred, x_true, path, a0):
    fig, ax = plt.subplots(figsize=(10, 5))
    r_values, g_r = compute_rdf(x_pred, 6*a0, 6*a0, 0.5*a0, a0/64)
    r_values_true, g_r_true = compute_rdf(x_true, 6*a0, 6*a0, 0.5*a0, a0/64)

    ax.plot(r_values.cpu(), g_r.cpu(), label='Predicted RDF')
    ax.plot(r_values_true.cpu(), g_r_true.cpu(), label='True RDF')
    ax.set_title('Radial Distribution Function')

    xticks = [0*a0, 1*a0, 2*a0, 3*a0, 4*a0, 5*a0, 6*a0]
    xtick_labels = ['0', 'a0', '2a0', '3a0', '4a0', '5a0', '6a0']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.legend()
    
    path = os.path.join(path, 'rdf.png')
    plt.savefig(path)
    plt.close()


def stress_LMC(cfg, sc, x, x_grid):
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


def set_seed(rank=0, seed=42):
    seed += rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)