import torch
from torch import nn
import torch.multiprocessing as mp

import shutil, os

# self-defined tools
from SAM import cfg, SEED
from SAM.train import train_model, train_model_ddp
from SAM.nn import ScoreNet, Unet
# from SAM.nn import marginal_prob_std, diffusion_coeff
from SAM.utils import Euler_Maruyama_sampler, pc_sampler
from SAM.utils import set_seed, SamDataset, MyDataset
from SAM.utils import stress_MD, stress_LMC, force
from SAM.data import generate_data, eq_D2Virial, process_data
from SAM.sde_lib import VESDE

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '64060'

if __name__ == "__main__":
    # =========================================== Score matching ===========================================
    set_seed(SEED)
    sample_all, x_ref_all, d2V, ind_sam = generate_data()
    sample_sam = sample_all[:, ind_sam, :]
    x_ref_sam = x_ref_all[ind_sam]
    dataset = SamDataset(sample_sam)
    # dataset = MyDataset(torch.transpose(sample_sam, 1, 2))
    score = ScoreNet(cfg.nd, hidden_depth=cfg.depth, embed_dim=cfg.width, use_bn=cfg.use_bn)
    # score = Unet(cfg.d)

    score.to(cfg.device)

    world_size = torch.cuda.device_count()
    args_train = (world_size, score, dataset, cfg.lr_train, cfg.batch_size//world_size, cfg.n_epochs, cfg.print_interval)

    if world_size == 1:
        train_model(score, dataset, cfg.lr_train, cfg.batch_size, cfg.n_epochs, cfg.print_interval, cfg.device)
    elif world_size >= 2:
        mp.spawn(train_model_ddp, args=args_train, nprocs=world_size)

    torch.save(score.state_dict(), cfg.path_model_output)
    shutil.copy(cfg.path_params, cfg.path_params_output)

    # =========================================== Sampling ===========================================
    sc_load = ScoreNet(cfg.nd, hidden_depth=cfg.depth, embed_dim=cfg.width, use_bn=cfg.use_bn)
    # sc_load = Unet(cfg.d)
    sc_load.to(cfg.device)
    sc_load.load_state_dict(torch.load(cfg.path_model_output, weights_only=True))

    shape = dataset.shape
    shape[0] = cfg.ntrajs_sample
    marginal_prob_std = lambda t : VESDE.marginal_prob_std(cfg.sigma_min, cfg.sigma_max, t)
    diffusion_coeff = lambda t : VESDE.diffusion_coeff(cfg.sigma_min, cfg.sigma_max, t)
    t = torch.ones(cfg.ntrajs_sample, device=cfg.device)
    init_x = torch.flatten(x_ref_sam) + torch.randn(shape, device=cfg.device) * marginal_prob_std(1.0)
    x_pred = Euler_Maruyama_sampler(sc_load, shape, init_x, diffusion_coeff, batch_size=cfg.ntrajs_sample, n_steps=cfg.n_steps, eps=cfg.eps)

    # =========================================== Data ===========================================
    # MD data
    sample_sam = sample_all[:, ind_sam, :]

    mean_true = torch.mean(sample_sam, axis=0)
    std_true = torch.std(sample_sam, axis=0)
    print('============== data ==============')
    # print('mean of true data:\n',mean_true.T,'\n')
    # print('coviriance of true data:\n',std_true.T,'\n')

    # LMC data
    x_ref_sam = x_ref_all[ind_sam]
    print('All data:\n',x_pred.shape[0],'\n')
    x_pred = process_data(x_pred, x_ref_sam)
    print('Processed data:\n',x_pred.shape[0],'\n')

    mean_lmc = torch.mean(x_pred, axis=0)
    std_lmc = torch.std(x_pred, axis=0)
    # print('mean of predicted data:\n',mean_lmc.view(cfg.n_sam, cfg.d).T,'\n')
    # print('coviriance of predicted data:\n',std_lmc.view(cfg.n_sam, cfg.d).T,'\n')

    # error
    err_mean = torch.mean((mean_true - mean_lmc)**2)
    err_std = torch.mean((std_true - std_lmc)**2)

    print('error of mean:\n',err_mean,'\n')
    print('error of coviriance:\n',err_std,'\n')
