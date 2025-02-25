import torch
import numpy as np
from model import utils


def get_optimizer(cfg, params):
    """Returns a flax optimizer object based on `config`."""
    if cfg.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, betas=(cfg.optim.beta1, 0.999), eps=cfg.optim.eps,
                                     weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS(
            params, lr=0.1, max_iter=10, max_eval=None, tolerance_grad=1e-09, tolerance_change=1e-11)
    else:
        raise NotImplementedError(
            f'Optimizer {cfg.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(cfg):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=cfg.optim.lr_train,
                    warmup=cfg.optim.warmup,
                    grad_clip=cfg.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


# ====================================   Loss Function   ==============================================================
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


def loss_ssm(model, samples, sigma=0.1):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    score = model(perturbed_samples)
    div_score = model.div(perturbed_samples)
    loss = torch.sum(score**2) + 2 * div_score
    return torch.mean(loss)


def get_sde_loss_fn(sde, train, reduce_mean=True, likelihood_weighting=False, eps=1e-5):

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """The loss function for training score-based generative models."""
        model_fn = utils.get_score_fn(sde, model, train)
        random_t = torch.rand(
            batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, random_t)
        perturbed_x = mean + z * std[:, None, None]
        score = model_fn(perturbed_x, random_t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), random_t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        return torch.mean(losses)

    return loss_fn


def get_step_fn(sde, train=True, optimizer=None, reduce_mean=False, likelihood_weighting=False):
    """Returns a step function based on `config`."""
    if train:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)
    else:
        pass

    def step_fn(state, batch):
        model = state['model']
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        state['step'] += 1
        return loss

    return step_fn
