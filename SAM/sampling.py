import torch
import time
from SAM import datasets, utils
from tqdm import tqdm
import math
import abc

_PREDICTORS = {}


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


def get_em_sampler(cfg, x_init, sde, sampling_eps):

    def Euler_Maruyama_sampler(score):
        score_fn = utils.get_score_fn(sde, score, train=True)

        time_steps = torch.linspace(1., sampling_eps, sde.N, device=cfg.device)
        dt = time_steps[1] - time_steps[0]
        x = x_init
        with torch.no_grad():
            for time_step in tqdm(time_steps, desc="Sampling"):
                t = torch.ones(cfg.sampler.ntrajs,
                               device=cfg.device) * time_step

                drift, diffusion = sde.sde(x, t)
                drift = drift - diffusion[:, None] ** 2 * score_fn(x, t)
                mean_x = x + drift * dt
                x = mean_x + torch.sqrt(-dt) * \
                    diffusion[:, None] * torch.randn_like(x)

        return mean_x

    return Euler_Maruyama_sampler


def get_sampling_fn(cfg, sde, sampling_eps):
    shape = (cfg.sampler.ntrajs, cfg.data.nd)
    x_ref_all, _, ind_sam = datasets.get_data_params(cfg)
    x_ref_sam = x_ref_all[ind_sam]
    x_init = sde.prior_sampling(shape, torch.flatten(x_ref_sam))

    if cfg.sampler.method == 'em':
        return get_em_sampler(cfg, x_init, sde, sampling_eps)
    elif cfg.sampler.method == 'pc':
        pass
    else:
        raise ValueError(f"Sampler {cfg.sampler} not recognized.")

# ====================================   Sampling   ==============================================================
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

#     print(f"=========================== Start Sampling ===========================")
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
#     print(f"=========================== Finished Sampling ===========================\n")
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

#     print(f"=========================== Start Sampling ===========================")
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
#     print(f"=========================== Finished Sampling ===========================\n")
#     return mean_x
