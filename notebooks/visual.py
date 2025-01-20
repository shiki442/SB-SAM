import torch
import sys

sys.path.append("..")
sys.path.append('../SAM')

from SAM.nn import ScoreNet, Unet
# from SAM.nn import marginal_prob_std, diffusion_coeff
from SAM.sde_lib import VESDE
from SAM.utils import Euler_Maruyama_sampler, pc_sampler
from SAM.utils import stress_LMC, stress_MD, force
from SAM.utils import set_seed
from SAM.datasets import generate_data, generate_PDC_data, process_data, eq_D2Virial

from matplotlib import pyplot as plt # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from SAM import cfg, SEED
sc_load = ScoreNet(cfg.nd, hidden_depth=cfg.depth, embed_dim=cfg.width, use_bn=cfg.use_bn)
# sc_load = Unet(cfg.d)

sc_load.to(cfg.device)
sc_load.load_state_dict(torch.load(cfg.path_params[:9] + 'models/sc' + cfg.path_params[22:-3] + 'pth', weights_only=True))

set_seed(SEED)
torch.set_printoptions(linewidth=1000, threshold=1000)

# generate dataset
# device
sample_all, x_ref_all, d2V, ind_sam = generate_data()
# sample_pdc, x_ref_pdc, d2V_pdc = generate_PDC_data()
sample_sam = sample_all[:, ind_sam, :]

# E-M sampling
shape = [1, cfg.nd]
x_ref_sam = x_ref_all[ind_sam]
marginal_prob_std = lambda t : VESDE.marginal_prob_std(cfg.sigma_min, cfg.sigma_max, t)
diffusion_coeff = lambda t : VESDE.diffusion_coeff(cfg.sigma_min, cfg.sigma_max, t)
t = torch.ones(cfg.ntrajs_sample, device=cfg.device)
init_x = torch.flatten(x_ref_sam) + torch.randn(shape, device=cfg.device) * marginal_prob_std(1.0)
x_pred = pc_sampler(sc_load, shape, init_x, diffusion_coeff, batch_size=cfg.ntrajs_sample, n_steps=cfg.n_steps, eps=0.01)

# =========================================== Data ===========================================
# MD data

mean_true = torch.mean(sample_sam, axis=0)
std_true = torch.std(sample_sam, axis=0)
# std_true = torch.cov(sample_test_true)
print('============== data ==============')
# print('mean of true data:\n',mean_true.T,'\n')
# print('coviriance of true data:\n',std_true.T,'\n')

# PDC data
# mean_pdc = torch.mean(sample_pdc, axis=0)
# std_pdc = torch.std(sample_pdc, axis=0)
# print('mean of pdc data:\n',mean_pdc.view(cfg.n_sam, cfg.d).T,'\n')
# print('coviriance of pdc data:\n',std_pdc.view(cfg.n_sam, cfg.d).T,'\n')

# LMC data
x_ref_sam = x_ref_all[ind_sam]
print('All data:\n',x_pred.shape[0],'\n')
x_pred = process_data(x_pred, x_ref_sam)
print('Processed data:\n',x_pred.shape[0],'\n')

mean_lmc = torch.mean(x_pred, axis=0)
std_lmc = torch.std(x_pred, axis=0)
# print('mean of predicted data:\n',mean_lmc.view(cfg.n_sam, cfg.d).T,'\n')
# print('coviriance of predicted data:\n',std_lmc.view(cfg.n_sam, cfg.d).T,'\n')

err_mean = torch.mean((mean_true - mean_lmc)**2)
err_std = torch.mean((std_true - std_lmc)**2)

print('error of mean:\n',err_mean,'\n')
print('error of coviriance:\n',err_std,'\n')
# =========================================== Calculate Stress ===========================================
# # MD stress
# eq_d2V = eq_D2Virial(d2V, ind_sam)
# # f = force(sample_sam, x_ref_sam, eq_d2V)
# f_MD = force(sample_all, x_ref_all, d2V)
# f_MD = f_MD[:,ind_sam,:]
# # f_sc = force(x_pred.cpu(), x_ref_sam, eq_d2V)
# Stress_MD, f1 = stress_MD(f_MD, x_ref_sam)
# # Stress_f_xsc, f2 = stress_MD(f_sc, x_ref_sam)

# # PDC stress
# f_PDC = force(sample_pdc, x_ref_pdc, d2V_pdc)
# Stress_PDC, f5 = stress_MD(f_PDC, x_ref_pdc)

# # LMC stress
# Stress_LMC_x, f3 = stress_LMC(sc_load, sample_sam, x_ref_sam)
# Stress_LMC_xsc, f4 = stress_LMC(sc_load, x_pred, x_ref_sam)

# =========================================== Print Output ===========================================
# f_mean = torch.mean(f1, axis=0)
# # f_xpred_mean = torch.mean(f2, axis=0)
# f_sc_mean = torch.mean(f3, axis=0)
# f_sc_xpred_mean = torch.mean(f4, axis=0)

# print('============== Force ==============')
# print(f'True Force on true samples: \n{f1[0].T}\n')
# # print(f'True Force on predicted samples: \n{f2[0].T}\n')
# print(f'Score on true samples: \n{f3[0].T}\n')
# print(f'Score on predicted samples: \n{f4[0].T}\n')

# print('============== Stress ==============')
# print(f'MD Stress :  P={Stress_MD.item():.5f}')
# # print(f'Stress with true f  and predicted x: P={Stress_f_xsc.item():.5f}')
# print(f'PDC Stress:  P={Stress_PDC.item():.5f}')
# print(f'LMC Stress:  P={Stress_LMC_x.item():.5f}')
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
# axes[0].plot(torch.arange(100), torch.abs(mean_true[:100, 0].cpu()-mean_lmc[:100, 0].cpu()))
# # axes[0].plot(torch.arange(100), 2*torch.ones(100), label='max mean = 2')
# axes[0].set_title(f'err of mean')
# # axes[0].legend()
# axes[0].set_ylim([0,0.1])
# # axes[0].set_xlabel('自由度')

# axes[1].plot(torch.arange(100), torch.abs(std_true[:100, 0].cpu()-std_lmc[:100, 0].cpu()))
# axes[1].set_title(f'err of std')
# axes[1].set_ylim([0,0.1])
# # axes[1].set_xlabel('自由度')

# plt.show()
# plt.close()

# sample_fig1 = sample_sam[:, 0, 0]
# axes[0].hist(sample_fig1.cpu(), bins=100, edgecolor='black', density=True)
# axes[0].set_title(f'true data: x1')

# sample_fig2 = x_pred[:, 0, 0]
# axes[1].hist(sample_fig2.cpu(), bins=100, edgecolor='black', density=True)
# axes[1].set_title(f'Prediction: x1')
# plt.show()
# plt.close()


# sample_fig1 = torch.square(sample_sam[:, 0, 0]) * sample_sam[:, 100, 0]
# axes[0].hist(sample_fig1.cpu(), bins=100, edgecolor='black', density=True)
# axes[0].set_title(f'true data: x1*x1*x101')

# sample_fig2 = torch.square(x_pred[:, 0, 0]) * x_pred[:, 100, 0]
# axes[1].hist(sample_fig2.cpu(), bins=100, edgecolor='black', density=True)
# axes[1].set_title(f'Prediction: x1*x1*x101')
# plt.show()
# plt.close()


sample_fig1 = torch.square(sample_sam[:, 3, 0]) - sample_sam[:, 4, 0]
axes[0].hist(sample_fig1.cpu(), bins=100, edgecolor='black', density=True)
axes[0].set_title(f'true data: x3*x3-x4')

sample_fig2 = torch.square(x_pred[:, 3, 0]) - x_pred[:, 4, 0]
axes[1].hist(sample_fig2.cpu(), bins=100, edgecolor='black', density=True)
axes[1].set_title(f'Prediction: x3*x3-x4')
plt.show()
plt.close()