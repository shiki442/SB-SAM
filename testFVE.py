import torch
import sys

sys.path.append("..")
sys.path.append('../SAM')

from SAM.utils import stress_LMC, stress_MD, force, potential
from SAM.utils import set_seed
from SAM.datasets import generate_data, generate_grid, index_sam, nearest_particles, D2Virial, generate_gauss_data, process_data, eq_D2Virial

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import torch

from SAM import cfg, SEED

set_seed(SEED)
torch.set_printoptions(linewidth=1000, threshold=1000, precision=6)


def get_mean_cov(n, mu=0., sigma=1., random=False):
    mean = mu*torch.ones(n)  # Mean vector
    if random == True:
        mean += torch.rand(n)
    cov = torch.zeros((cfg.data.d, n, n))
    diag_vals = [10., -9., 8., -7., 6., -5.]
    # 构造三对角矩阵
    for i in range(len(diag_vals)):
        if i <= n:
            diag = diag_vals[i] * torch.ones(n - i)
            cov[0] = cov[0] + torch.diag(diag, diagonal=i) + torch.diag(diag, diagonal=-i)
    return mean, cov


# generate dataset
# sample_all, x_ref_all, d2V, ind_sam = generate_data()
d = 1
n_list = [5,10,15,20,25,30,35,40,45,50]
N = 50
k_nearest = [5,10,15,20]

# d = 3
# n_list = [4,8,12,16,20]
# k_nearest = [5,10,15,20]

V_list = [[0]*len(n_list)]*len(k_nearest)
Veq_list = [[0]*len(n_list)]*len(k_nearest)

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

for j in range(len(k_nearest)):
    print("k_nearest=", k_nearest[j])
    for i in range(len(n_list)):
        n = n_list[i]
        m = k_nearest[j]
        """Generate a dataset from the example problem."""
        min_grid = -0.5 * 0.1 * n
        max_grid = 0.5 * 0.1 * n
        
        x_ref_all = generate_grid(d, n, min_grid, max_grid)
        indij_ref = generate_grid(d, n, mode='index')
        indij_near = nearest_particles(indij_ref, m, d)
        d2V = D2Virial(indij_ref, indij_near, n, d)
        sample_all = generate_gauss_data(x_ref_all, d2V, cfg)
        mean, cov = get_mean_cov(n)
        V_list[j][i] = potential(sample_all, x_ref_all, cov)

        n_all_per_dim = N + 1
        x_ref_all = generate_grid(d, n_all_per_dim, -0.5 * 0.1 * N, 0.5 * 0.1 * N)
        indij_ref = generate_grid(d, n_all_per_dim, mode='index')
        indij_near = nearest_particles(indij_ref, m, d)
        d2V = D2Virial(indij_ref, indij_near, n_all_per_dim, d)
        ind_sam = index_sam(x_ref_all, min_grid, max_grid)
        x_ref_sam = x_ref_all[ind_sam]
        eq_d2V = eq_D2Virial(d2V, ind_sam, d)
        mean, cov = get_mean_cov(len(ind_sam))
        sample_eq_sam = generate_gauss_data(x_ref_sam, eq_d2V, cfg)
        # Deq = torch.linalg.eig(eq_d2V[0])
        # print(Deq)
        
        Veq_list[j][i] = potential(sample_eq_sam, x_ref_sam, cov)

    axes[0].plot(n_list, V_list[j], marker='o', label=f"{k_nearest[j]} nearest neighbors atoms")
    axes[1].plot(n_list, Veq_list[j], marker='o', label=f"{k_nearest[j]} nearest neighbors atoms")

axes[0].set_title(f'd={d},y='r'$\mathbb{E}u^{T}u/n$')
axes[0].legend()
axes[0].set_xlim([-1,55])
axes[1].set_title(f'd={d},y='r'$\mathbb{E}u^{T}u/n$')
axes[1].legend()
plt.show()
