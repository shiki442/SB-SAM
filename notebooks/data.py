import torch
import numpy as np

n_out = 2000

def index_sam(ref_struc, min_x, max_x, defm):
    defm_tensor = torch.tensor(defm, device=ref_struc.device)
    inv_defm = torch.linalg.inv(defm_tensor)
    ref_struc[0] = inv_defm @ ref_struc[0]
    eps = 1.0e-3 * (ref_struc[0, 0, 1] - ref_struc[0, 0, 0])
    # 这里是max_x+eps
    mask = (ref_struc >= min_x-eps) & (ref_struc <= max_x+eps)
    mask = mask.all(dim=1)
    index_sam = torch.nonzero(mask[0]).squeeze()
    return index_sam

a0 = 2.86449264163000

data_file = 'C:/Users/78103/OneDrive/PC/Code Library/SAM_dataset/data5/100_01/pos-f.dat'
data = np.loadtxt(data_file, dtype=np.float32)

data = data.reshape(-1, 250, 6)
data = torch.from_numpy(data[:, :, :3]).permute(0, 2, 1)
data = data[0:n_out]


pos_path = 'C:/Users/78103/OneDrive/PC/Code Library/SAM_dataset/data5/100_01/init_pos.dat'
pos = np.loadtxt(pos_path, dtype=np.float32)
pos = pos.reshape(1, 2*pos.shape[0], 3)
x0 = torch.from_numpy(pos).permute(0, 2, 1)
defm = torch.eye(3, device=x0.device, dtype=x0.dtype)
x0[0] = defm @ x0[0]

ind_sam = index_sam(x0, 2*a0, 3*a0, defm)

data0 = data[:, :, ind_sam]
id = torch.zeros(1, 1, pos.shape[1])
id[:,:,ind_sam] = 1
output = torch.cat((x0, id), dim=1)
output = output.repeat(n_out, 1, 1)
output = torch.cat((data, output), dim=1)
output_np = output.permute(0, 2, 1).cpu().numpy()
for i in range(n_out):
    path_output = f'D:/CodeLibrary/SAM_dataset/stress/xyz_{i}.dat'
    np.savetxt(path_output, output_np[i])

print(data0.shape)