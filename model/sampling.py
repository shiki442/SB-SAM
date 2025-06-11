import torch
import time
from model import datasets, utils
from tqdm import tqdm
import math
import abc

_PREDICTORS = {}
_CORRECTORS = {}

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor."""
        pass

class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector."""
    pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * math.sqrt(-dt) * z
        return x, x_mean

def get_pc_sampler(cfg, x_init, sde, sampling_eps, predictor, corrector):
    
    def pc_sampler(model):
        """ The PC sampler funciton.

        Args:
        model: A score model.
        Returns:
        Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            x, x_mean = corrector_update_fn(x, vec_t, model=model)
            x, x_mean = predictor_update_fn(x, vec_t, model=model)

        return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)
    
    return pc_sampler

def get_em_sampler(cfg, x_init, sde, sampling_eps):
    if cfg.world_size > 1:
        n = cfg.sampler.ntrajs // cfg.world_size
    else:
        n = cfg.sampler.ntrajs
    cond = datasets.get_condition(cfg, n)

    def Euler_Maruyama_sampler(score):
        score.eval()
        score_fn = utils.get_score_fn(sde, score, train=True)

        time_steps = torch.linspace(1., sampling_eps, sde.N, device=cfg.device)
        dt = time_steps[1] - time_steps[0]
        x = x_init
        tqdm_bar = tqdm(time_steps, desc="Sampling", mininterval=2, ncols=0)
        eta = 0.2
        with torch.no_grad():
            for time_step in tqdm_bar:
                t = torch.ones(n, device=cfg.device) * time_step
                drift, diffusion = sde.sde(x, t)
                drift = drift - diffusion[:, None, None] ** 2 * ((1+eta) * score_fn(x, t, cond) - eta * score_fn(x, t))
                mean_x = x + drift * dt
                x = mean_x + torch.sqrt(-dt) * \
                    diffusion[:, None, None] * torch.randn_like(x)

        return mean_x

    return Euler_Maruyama_sampler

def get_langevin_sampler(cfg, x_init, sde, sampling_eps):
    if cfg.world_size > 1:
        n = cfg.sampler.ntrajs // cfg.world_size
    else:
        n = cfg.sampler.ntrajs
    tau = 0.01 * torch.tensor(cfg.eval.temperature, device=cfg.device)
    tau = tau.unsqueeze_(0).repeat(n, 1)
    stress = torch.tensor(cfg.data.defm, device=cfg.device)
    stress = 100 * (stress - torch.eye(3,3, device=stress.device))
    stress = stress[0, 0:2].repeat(n, 1)
    cond = torch.cat((tau, stress), dim=1)

    def Langevin_sampler(score):
        score.eval()
        score_fn = utils.get_score_fn(sde, score, train=True)

        time_steps = torch.linspace(1., sampling_eps, sde.N, device=cfg.device)
        dt = time_steps[1] - time_steps[0]
        x = x_init
        tqdm_bar = tqdm(time_steps, desc="Sampling", mininterval=2, ncols=0)
        with torch.no_grad():
            for time_step in tqdm_bar:
                t = torch.ones(n, device=cfg.device) * time_step
                grad = score_fn(x, t, cond)
                grad_norm = torch.norm(grad, dim=-1).mean()
                noise_norm = math.sqrt(x.shape[1])
                langevin_step_size = 2 * (cfg.sampler.snr * noise_norm / grad_norm)**2
                mean_x = x + langevin_step_size * grad 
                x = mean_x + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
                # drift, diffusion = sde.sde(x, t)
                # mean_x = x + drift * dt
                # x = mean_x + torch.sqrt(-dt) * \
                #     diffusion[:, None, None] * torch.randn_like(x)

        return mean_x

    return Langevin_sampler

def get_sampling_fn(cfg, sde, sampling_eps):
    x0 = datasets.get_init_pos(cfg, mode='eval')
    ind = datasets.index_sam(x0, cfg.data.min_x, cfg.data.max_x, cfg.data.defm)
    shape = (cfg.sampler.ntrajs//cfg.world_size, cfg.data.d, cfg.data.n)
    x_init = sde.prior_sampling(shape, x0[:,:,ind])

    if cfg.sampler.method == 'em':
        return get_em_sampler(cfg, x_init, sde, sampling_eps)
    elif cfg.sampler.method == 'langevin':
        return get_langevin_sampler(cfg, x_init, sde, sampling_eps)
    elif cfg.sampler.method == 'pc':
        predictor = _PREDICTORS[cfg.sampling.predictor.lower()]
        corrector = _CORRECTORS[cfg.sampling.corrector.lower()]
        sampling_fn = get_pc_sampler(cfg, x_init, sde, sampling_eps, predictor, corrector)
    else:
        raise ValueError(f"Sampler {cfg.sampler} not recognized.")

# ===================================   Sampling   ============================================================
# def Euler_Maruyama_sampler(score_model,
#                            shape,
#                            x_init,
#                            diffusion_coeff,
#                            batch_size=64,
#                            n_steps=1000,
#                            eps=1e-3):
#     start_time = time.time()

#     t = torch.ones(batch_size, device=cfg.device)
#     # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None]
#     # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None, None] # Unet
#     time_steps = torch.linspace(1., eps, n_steps, device=cfg.device)
#     step_size = time_steps[0] - time_steps[1]
#     x = x_init

#     print(f"========================== Start Sampling ==========================")
#     with torch.no_grad():
#         for time_step in tqdm(time_steps, desc="Sampling"):
#             batch_time_step = torch.ones(batch_size, device=cfg.device) * time_step
#             g = diffusion_coeff(batch_time_step)[:, None]
#             # g = diffusion_coeff(batch_time_step)[:, None, None] # Unet
#             mean_x = x + (g**2) * score_model(x, batch_time_step) * step_size
#             x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)

#     end_time = time.time()
#     print(f"Num of trajectories: {batch_size}")
#     print(f"Num of SDE Steps: {n_steps}")
#     print(f"Smallest time step: {eps}")
#     print(f"Total time = {(end_time-start_time)/60.:.5f}m")
#     print(f"========================== Finished Sampling ==========================\n")
#     return mean_x


# def pc_sampler(score_model,
#                 shape,
#                 init_x,
#                 diffusion_coeff,
#                 batch_size=64,
#                 n_steps=1000,
#                 snr=0.01,
#                 eps=1e-3):
#     start_time = time.time()

#     t = torch.ones(batch_size, device=cfg.device)
#     shape[0] = batch_size
#     # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None]
#     # init_x = torch.randn(shape, device=cfg.device) * marginal_prob_std(t)[:, None, None] # Unet
#     time_steps = torch.linspace(1., eps, n_steps, device=cfg.device)
#     step_size = time_steps[0] - time_steps[1]
#     x = init_x

#     print(f"========================== Start Sampling ==========================")
#     with torch.no_grad():
#         for time_step in tqdm(time_steps, desc="Sampling"):
#             batch_time_step = torch.ones(batch_size, device=cfg.device) * time_step

#             # Corrector step (Langevin MCMC)
#             grad = score_model(x, batch_time_step)
#             grad_norm = torch.norm(grad, dim=-1).mean()
#             # grad_norm = torch.tensor(25.0)
#             noise_norm = math.sqrt(x.shape[1])
#             langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
#             x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
#             # Predictor step (Euler-Maruyama)
#             g = diffusion_coeff(batch_time_step)[:, None]
#             # g = diffusion_coeff(batch_time_step)[:, None, None] # Unet
#             mean_x = x + (g**2) * score_model(x, batch_time_step) * step_size
#             x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)

#     end_time = time.time()
#     print(f"Num of trajectories: {batch_size}")
#     print(f"Num of SDE Steps: {n_steps}")
#     print(f"Smallest time step: {eps}")
#     print(f"Total time = {(end_time-start_time)/60.:.5f}m")
#     print(f"========================== Finished Sampling ==========================\n")
#     return mean_x
