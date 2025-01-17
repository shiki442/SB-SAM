import torch
import time
import math
from SAM import cfg
from SAM.utils import force

###====================================   Dataset   ==============================================================
def generate_gauss_data(mean, d2V, d, T):
    x_all = torch.empty([cfg.ntrajs_train, mean.shape[0], d])
    # mean = torch.zeros_like(mean)
    cov = torch.empty_like(d2V)
    for i in range(d):
        cov[i] = T/2*torch.linalg.inv(d2V[i])
        sampler  = torch.distributions.MultivariateNormal(mean[:, i], cov[i])
        x_all[:,:,i] = sampler.sample((cfg.ntrajs_train, ))
    return x_all


def get_mean_cov(mu=0., sigma=1., random=False):
    mean = mu*torch.ones(cfg.n_all)  # Mean vector
    if random == True:
        mean += torch.rand(cfg.n_all)
    cov = torch.zeros((cfg.n_all, cfg.n_all))
    diag_vals = [sigma/2., -0.5, 0.02, -0.01, -0.005]
    # 构造三对角矩阵
    for i in range(5):
        if i <= cfg.n_all:
            diag = diag_vals[i] * torch.ones(cfg.n_all - i)
            cov = cov + torch.diag(diag, diagonal=i) + torch.diag(diag, diagonal=-i)
    return mean, cov


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


def index_sam(ref_struc, min_sam=cfg.min_sam, max_sam=cfg.max_sam):
    mask = (ref_struc >= min_sam) & (ref_struc <= max_sam)
    mask = mask.all(dim=1)
    index_sam = torch.nonzero(mask).squeeze()
    return index_sam


def nearest_particles(r, dist_max, d=cfg.d):
    r = generate_grid(d, math.ceil(dist_max), mode='relative_index')
    dist = torch.sqrt(torch.sum(torch.square(r), dim=1))
    ind = (dist <= dist_max) & (dist > 0.0)
    return r[ind]


def D2Virial(indij_ref, indij_near, n, d=cfg.d):
    n_all = n ** d
    d2V = torch.zeros([d, n_all, n_all])
    ind_x = torch.arange(0, n_all)
    indij_y = torch.zeros_like(indij_ref)
    for i in range(d):
        # The case of r=0
        d2V_ij = torch.zeros([n_all, n_all])
        d2V_ij[ind_x, ind_x] += Hooke_coeff(torch.zeros_like(indij_near[0]))
        # The case of r!=0
        for r in indij_near:
            indij_y = indij_ref + r
            indij_y = (indij_y + n) % n
            ind_y = torch.zeros_like(ind_x)
            for k in range(indij_y.shape[1]):
                ind_y += n**k * indij_y[:,-(k+1)]
            d2V_ij[ind_x, ind_x] += Hooke_coeff(r)
            d2V_ij[ind_x, ind_y] -= Hooke_coeff(r)
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
        D11 = d2V[i][ind_sam][:,ind_sam]
        D12 = d2V[i][ind_sam][:,ind_other]
        D22 = d2V[i][ind_other][:,ind_other]
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


def Hooke_coeff(r):
    k0 = cfg.k0
    dist = torch.sqrt(torch.sum(torch.square(r)))
    if dist <= 1e-3:
        dist = 2.0
    return k0 / dist


def process_data(x_pred, x_ref_sam):
    if x_pred.dim() == 2:
        mu = torch.flatten(x_ref_sam)
    else:
        mu = x_ref_sam.T
    is_outliers = (torch.abs(x_pred-mu) >= cfg.grid_step)
    # tolerance = max(cfg.max_grid, -cfg.min_grid) + cfg.grid_step
    # is_outliers = (torch.abs(x_pred) >= tolerance)
    outliers = torch.where(torch.isnan(x_pred) | torch.isinf(x_pred) | is_outliers)
    rows_to_delete = torch.unique(outliers[0])
    n_pred = cfg.ntrajs_sample - len(rows_to_delete)
    mask = torch.ones(x_pred.size(0), dtype=torch.bool)
    mask[rows_to_delete] = False
    x_pred = x_pred[mask]
    x_pred_p = x_pred.view(n_pred, cfg.n_sam, cfg.d)
    return x_pred_p


def generate_data():
    """Generate a dataset from the example problem."""
    print(f"=========================== Starting data generation ===========================")
    start_time = time.time()

    x_ref_all = generate_grid(cfg.d, cfg.n_all_per_dim, cfg.min_grid, cfg.max_grid)
    indij_ref = generate_grid(cfg.d, cfg.n_all_per_dim, mode='index')
    ind_sam = index_sam(x_ref_all)
    indij_near = nearest_particles(indij_ref, cfg.d_max)
    d2V = D2Virial(indij_ref, indij_near, cfg.n_all_per_dim)
    sample_all = generate_gauss_data(x_ref_all, d2V, cfg.d, cfg.T)

    end_time = time.time()
    print(f"Dataset Size: {sample_all.shape}")
    print(f"Dim: {cfg.d}")
    print(f"Num of Total Particles: {cfg.n_all}")
    print(f"Num of SB-SAM Particles: {cfg.n_sam}")
    print(f"Total time = {(end_time-start_time)/60.:.5f}m")
    print(f"=========================== Finished data generation  ===========================\n")

    return sample_all.to(cfg.device), x_ref_all.to(cfg.device), d2V.to(cfg.device), ind_sam.to(cfg.device)


def generate_PDC_data():
    """Generate a dataset from the example problem."""
    print(f"=========================== Starting data generation ===========================")
    start_time = time.time()

    x_ref_sam = generate_grid(cfg.d, cfg.n_sam_per_dim, cfg.min_sam, cfg.max_sam)
    indij_ref = generate_grid(cfg.d, cfg.n_sam_per_dim, mode='index')
    indij_near = nearest_particles(indij_ref, cfg.d_max)
    d2V_PDC = D2Virial(indij_ref, indij_near, cfg.n_sam_per_dim)
    sample_sam = generate_gauss_data(x_ref_sam, d2V_PDC, cfg.d, cfg.T)

    end_time = time.time()
    print(f"Dataset Size: {sample_sam.shape}")
    print(f"Dim: {cfg.d}")
    print(f"Num of Total Particles: {cfg.n_all}")
    print(f"Num of SB-SAM Particles: {cfg.n_sam}")
    print(f"Total time = {(end_time-start_time)/60.:.5f}m")
    print(f"=========================== Finished data generation  ===========================\n")

    return sample_sam, x_ref_sam, d2V_PDC