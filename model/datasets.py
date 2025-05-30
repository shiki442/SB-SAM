import torch
from torch.utils.data import Dataset
import numpy as np

import time
import os
import math
from pathlib import Path


class PdcDataset(Dataset):
    def __init__(self, x_all):
        super().__init__()
        self.x_all = x_all
        self.mean = torch.mean(self.x_all, axis=0)
        self.std = torch.std(self.x_all, axis=0)

    def __len__(self):
        return self.x_all.shape[0]

    def __getitem__(self, idx):
        return torch.flatten(self.x_all[idx], start_dim=-2)


class SamDataset(Dataset):
    def __init__(self, x_all, cond=None, ind_sam=None):
        super().__init__()
        self.x_all = x_all
        self.x_sam = x_all[:, :, ind_sam]
        self.cond = cond[:,:3]
        self.ind_sam = ind_sam
        self.shape = x_all.shape
        self.mean = torch.mean(self.x_sam, axis=0)
        self.std = torch.std(self.x_sam, axis=0)

    def __len__(self):
        return self.x_all.shape[0]

    def __getitem__(self, idx):
        return self.x_sam[idx], self.cond[idx]

# ===================================   Dataset   ============================================================


def get_dataset(cfg, mode):
    if cfg.data.problem == 'quadratic_potential':
        sample_all, ind_sam = quadratic_potential_data(cfg)
    elif cfg.data.problem == 'fe211':
        sample_all, cond, ind_sam = read_md_data(cfg, mode)
    dataset = SamDataset(sample_all, cond, ind_sam)
    return dataset


def get_pdc_dataset(cfg):
    sample_all = generate_pdc_data(cfg)
    dataset = PdcDataset(sample_all)
    return dataset


def generate_gauss_data(mean, d2V, cfg):
    d = cfg.data.d
    T = cfg.eval.temperature
    x_all = torch.empty([cfg.training.ntrajs, d, mean.shape[0]])
    # mean = torch.zeros_like(mean)
    cov = torch.empty_like(d2V)
    for i in range(d):
        cov[i] = T/2*torch.linalg.inv(d2V[i])
        sampler = torch.distributions.MultivariateNormal(mean[:, i], cov[i])
        x_all[:, i, :] = sampler.sample((cfg.training.ntrajs, ))
    return x_all


def symmetrize_data(x):
    """Symmetrize the data."""
    symmetrized = torch.concat((x, -x), axis=0)
    return symmetrized


def generate_grid(dim, n, min_val=-0.0, max_val=1.0, mode='pos'):
    if mode == 'pos':
        ranges = [torch.linspace(min_val, max_val, n) for _ in range(dim)]
    elif mode == 'index':
        ranges = [torch.arange(0, n) for _ in range(dim)]
    elif mode == 'relative_index':
        ranges = [torch.arange(-n, n+1) for _ in range(dim)]
    grids = torch.meshgrid(*ranges, indexing='ij')
    flattened_grid = torch.stack(grids).view(dim, -1).T
    return flattened_grid


def index_sam(ref_struc, min_x, max_x, defm):
    defm_tensor = torch.tensor(defm, device=ref_struc.device)
    inv_defm = torch.linalg.inv(defm_tensor)
    ref_struc[0] = inv_defm @ ref_struc[0]
    eps = 1.0e-4 * (ref_struc[0, 0, 1] - ref_struc[0, 0, 0])
    mask = (ref_struc >= min_x-eps) & (ref_struc <= max_x-eps)
    mask = mask.all(dim=1)
    index_sam = torch.nonzero(mask[0]).squeeze()
    return index_sam


def nearest_particles(r, dist_max, d):
    r = generate_grid(d, math.ceil(dist_max), mode='relative_index')
    dist = torch.sqrt(torch.sum(torch.square(r), dim=1))
    ind = (dist <= dist_max) & (dist > 0.0)
    return r[ind]


def D2Virial(indij_ref, indij_near, cfg):
    n = cfg.data.nx_max
    d = cfg.data.d
    n_max = n ** d
    d2V = torch.zeros([d, n_max, n_max])
    ind_x = torch.arange(0, n_max)
    indij_y = torch.zeros_like(indij_ref)
    Hooke_coeff_fn = get_Hooke_coeff_fn(cfg)
    for i in range(d):
        # The case of r=0
        d2V_ij = torch.zeros([n_max, n_max])
        d2V_ij[ind_x, ind_x] += Hooke_coeff_fn(torch.zeros_like(indij_near[0]))
        # The case of r!=0
        for r in indij_near:
            indij_y = indij_ref + r
            indij_y = (indij_y + n) % n
            ind_y = torch.zeros_like(ind_x)
            for k in range(indij_y.shape[1]):
                ind_y += n**k * indij_y[:, -(k+1)]
            d2V_ij[ind_x, ind_x] += Hooke_coeff_fn(r)
            d2V_ij[ind_x, ind_y] -= Hooke_coeff_fn(r)
        # symmetrize
        d2V_ij = 0.5 * (d2V_ij + d2V_ij.T)
        d2V[i] += d2V_ij
    return d2V


def eq_D2Virial(d2V, ind_sam, d):
    n = len(ind_sam)
    all_ind = torch.arange(d2V[0].shape[0])
    mask = torch.ones_like(all_ind, dtype=bool)
    mask[ind_sam] = False
    ind_other = all_ind[mask]
    eq_d2V = torch.zeros([d2V.shape[0], n, n])
    for i in range(d):
        D11 = d2V[i][ind_sam][:, ind_sam]
        D12 = d2V[i][ind_sam][:, ind_other]
        D22 = d2V[i][ind_other][:, ind_other]
        inv_D22 = torch.linalg.inv(D22)
        eq_d2V[i] = D11 - D12 @ inv_D22 @ D12.T
    return eq_d2V


def N(r: torch.Tensor):
    '''The number of particles with a relative position of |r|.'''
    d = len(r)
    n = math.factorial(d)
    _, counts = r.unique(return_counts=True)
    equal_counts = counts[counts > 1]
    for i in range(len(equal_counts)):
        n /= math.factorial(equal_counts[i])
    zero_count = torch.sum(r == 0).item()
    return n * 2**(d-zero_count)


def get_Hooke_coeff_fn(cfg):
    def Hooke_coeff(r):
        k0 = cfg.dynamics.k0 / cfg.data.grid_step**2
        dist = torch.sqrt(torch.sum(torch.square(r)))
        if dist <= 1e-3:
            dist = 2.0
        return k0 / dist
    return Hooke_coeff


def process_data(x_pred, x_ref_sam, cfg):
    if x_pred.dim() == 2:
        mu = torch.flatten(x_ref_sam)
    else:
        mu = x_ref_sam
    is_outliers = (torch.abs(x_pred-mu) >= cfg.data.grid_step)
    # tolerance = max(cfg.data.max_grid, -cfg.data.min_grid) + cfg.data.grid_step
    # is_outliers = (torch.abs(x_pred) >= tolerance)
    outliers = torch.where(torch.isnan(
        x_pred) | torch.isinf(x_pred) | is_outliers)
    rows_to_delete = torch.unique(outliers[0])
    n_pred = cfg.sampler.ntrajs - len(rows_to_delete)
    mask = torch.ones(x_pred.size(0), dtype=torch.bool)
    mask[rows_to_delete] = False
    x_pred = x_pred[mask]
    # x_pred_p = x_pred.view(n_pred, cfg.data.n, cfg.data.d)
    return x_pred


def get_init_pos(cfg):
    if cfg.data.problem == 'quadratic_potential':
        x_ref_all, ind_sam, _ = get_quadra_data_params(cfg)
        x_ref_sam = x_ref_all[ind_sam]
        x0 = x_ref_sam.T[None, :]
        return x0
    elif cfg.data.problem == 'fe211':
        pos = np.loadtxt(cfg.data.eval_data_dir + 'init_pos.dat', dtype=np.float32)
        pos = pos.reshape(1, 2*pos.shape[0], cfg.data.d)
        x0 = torch.from_numpy(pos).permute(0, 2, 1)
        defm = torch.tensor(cfg.data.defm)
        x0[0] = defm @ x0[0]
        return x0.to(cfg.device)


def get_quadra_data_params(cfg):
    x_ref_all = generate_grid(
        cfg.data.d, cfg.data.nx_max, cfg.data.min_grid, cfg.data.max_grid)
    indij_ref = generate_grid(cfg.data.d, cfg.data.nx_max, mode='index')
    ind_sam = index_sam(x_ref_all, cfg.data.min_x, cfg.data.max_x, cfg.data.defm)
    indij_near = nearest_particles(indij_ref, cfg.dynamics.k_near, cfg.data.d)
    d2V = D2Virial(indij_ref, indij_near, cfg)
    return x_ref_all.to(cfg.device), ind_sam, d2V.to(cfg.device)


def get_eq_quadra_data_params(cfg):
    x_ref_all, ind_sam, d2V = get_quadra_data_params(cfg)
    x_ref_sam = x_ref_all[ind_sam]
    d2V_eq = eq_D2Virial(d2V, ind_sam, cfg.data.d)
    return x_ref_sam, d2V_eq.to(cfg.device)


def quadratic_potential_data(cfg):
    """Generate a dataset from the example problem."""
    if cfg.log.verbose:
        print(f"========================== Starting data generation ==========================")
        start_time = time.time()

    x_ref_all, ind_sam, d2V = get_quadra_data_params(cfg)
    sample_all = generate_gauss_data(x_ref_all, d2V, cfg)
    if cfg.log.verbose:
        end_time = time.time()
        print(f"Dataset Size: {sample_all.shape}")
        print(f"Dim: {cfg.data.d}")
        print(f"Num of Total Particles: {cfg.data.n_max}")
        print(f"Num of SB-SAM Particles: {cfg.data.n}")
        print(f"Total time = {(end_time-start_time)/60.:.5f}m")
        print(f"========================== Finished data generation  ==========================\n")

    return sample_all.to(cfg.device), ind_sam


def generate_pdc_data(cfg):
    """Generate a dataset from the example problem."""
    if cfg.log.verbose:
        print(f"========================== Starting data generation ==========================")
        start_time = time.time()

    x_ref_all, d2V_eq = get_eq_quadra_data_params(cfg)
    sample_all = generate_gauss_data(x_ref_all, d2V_eq, cfg)

    if cfg.log.verbose:
        end_time = time.time()
        print(f"Dataset Size: {sample_all.shape}")
        print(f"Dim: {cfg.data.d}")
        print(f"Num of Total Particles: {cfg.data.n_max}")
        print(f"Total time = {(end_time-start_time)/60.:.5f}m")
        print(f"========================== Finished data generation  ==========================\n")

    return sample_all.to(cfg.device)


def fe211_md_data(cfg):
    data_dir = cfg.data.train_data_dir
    output_file = os.path.join(data_dir, 'fe211_md.pt')
    if 'fe211_md.pt' in os.listdir(data_dir):
        sample_all = torch.load(
            output_file, weights_only=True, map_location=cfg.device)
    else:
        file_list = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.dat')])

        all_data = []
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            data = np.loadtxt(file_path, dtype=np.float32)
            all_data.append(data[:, :3])

        all_data = np.stack(all_data)
        sample_all = torch.from_numpy(all_data).permute(0, 2, 1)

        torch.save(sample_all, output_file)
    ind_sam = torch.arange(0, sample_all.shape[-1])
    rawdata_dir = Path(data_dir)
    if cfg.data.delete_rawdata:
        for file in rawdata_dir.glob("*.dat"):
            os.remove(file)
    return sample_all, ind_sam


def read_md_data(cfg, mode='train'):
    all_data = []
    all_cond = []
    if mode == 'train':
        data_dir = cfg.data.train_data_dir
        data_dir_list = os.listdir(data_dir)
    elif mode == 'eval':
        data_dir = ''
        data_dir_list = [cfg.data.eval_data_dir]

    for subdir in data_dir_list:
        data_file = os.path.join(data_dir, subdir, 'pos-f.dat')
        params_file = os.path.join(data_dir, subdir, 'data_params.dat')
        output_file = os.path.join(data_dir, subdir, 'fe211_md.pt')
        if 'fe211_md.pt' in os.listdir(os.path.join(data_dir, subdir)):
            data = torch.load(
                output_file, weights_only=True, map_location=cfg.device)
            all_data.append(data['samples'])
            all_cond.append(data['conds'])
        else:
            data = np.loadtxt(data_file, dtype=np.float32)
            data = data.reshape(-1, cfg.data.n_max, 2*cfg.data.d)
            data = torch.from_numpy(
                data[:, :, :cfg.data.d]).permute(0, 2, 1).to(cfg.device)

            temperature = read_temperature(params_file)
            tau = 0.01 * torch.tensor(temperature, device=cfg.device)
            tau = tau.unsqueeze_(0).repeat(data.shape[0], 1)
            stress = read_stress(params_file)
            stress = torch.tensor(stress, device=cfg.device)
            stress = stress.unsqueeze_(0).repeat(data.shape[0], 1)
            cond = torch.cat((tau, stress), dim=1)

            all_data.append(data)
            all_cond.append(cond)
            torch.save({'samples': data, 'conds': cond}, output_file)
    x_ref = get_init_pos(cfg)
    ind_sam = index_sam(x_ref, cfg.data.min_x, cfg.data.max_x, cfg.data.defm)
    all_data = torch.cat(all_data)
    all_cond = torch.cat(all_cond)

    return all_data[:cfg.training.ntrajs], all_cond[:cfg.training.ntrajs], ind_sam


def read_temperature(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming the 10th line contains the Temperature data
        temperature_line = lines[9]
        # Split the line by comma and strip any extra whitespace
        temperature_value = temperature_line.split(',')[1].strip()
        return float(temperature_value)

def read_stress(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming the 11th line contains the Stress data
        stress_lines = lines[12:15]
        # Split the line and strip any extra whitespace
        stress_values = []
        for line in stress_lines:
            stress_values.extend([float(val) for val in line.split()])
        return stress_values

if __name__ == "__main__":
    fe211_md_data('./data')
