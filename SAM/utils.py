# Practical tools used by main.py

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import sys, time
import math
import random

sys.path.append("..")
sys.path.append("./project/songpengcheng/SAM_torch")

from SAM import sde_lib

###====================================   Set Seed   ==============================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_MODELS = {}

def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
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
    
    def score_fn(x, t):
        score = model(x, t)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = -score / std[:, None]
        return score
    
    return score_fn

###====================================   Stress   ==============================================================
def potential(x_all, mean, d2V):
    u = x_all - mean
    V = 0.
    for i in range(1):
        f = u[:,:,i] @ d2V[i]
        V += 0.5 * torch.mean(f * u[:,:,i])
        # V += torch.mean(torch.pow(u, 4))
    return V

def force(cfg, x_all, mean, d2V):
    f = torch.zeros_like(x_all)
    for i in range(cfg.d):
        f[:,:,i] = 2*(x_all[:,:,i] - mean[:, i]) @ d2V[i]
    return f

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
