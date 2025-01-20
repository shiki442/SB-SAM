import torch
import numpy as np
from SAM import utils

def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config.lr_train, betas=(config.beta1, 0.999), eps=config.eps,
                            weight_decay=config.weight_decay)
    elif config.optim.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS(params, lr=0.1, max_iter=10, max_eval=None, tolerance_grad=1e-09, tolerance_change=1e-11)
    else:
        raise NotImplementedError(
        f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer

def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr_train,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


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

def get_sde_loss_fn(sde, train, eps=1e-5):

    def loss_fn(model, batch):
        """The loss function for training score-based generative models."""
        model_fn = utils.get_score_fn(sde, model, train)
        random_t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps  
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, random_t)
        perturbed_x = mean + z * std[:, None]
        score = model_fn(perturbed_x, random_t)
        losses = torch.square(score * std[:, None] + z)
        return torch.mean(losses)
    
    return loss_fn


def get_step_fn(sde, train=True, optimizer=None):
    """Returns a step function based on `config`."""
    if train:
        loss_fn = get_sde_loss_fn(sde, train)
    else:
        pass
    
    def step_fn(model, batch):
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        return loss

    return step_fn