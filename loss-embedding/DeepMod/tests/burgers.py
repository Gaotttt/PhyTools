# Burgers, tests 1D input

# General imports
import sys
import warnings
from os.path import join as ospj
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
sys.path.insert(1, '../..')
from src.deepymod_torch.library_functions import library_1D_in
from src.deepymod_torch.DeepMod import DeepMod
from src.deepymod_torch.training import train_deepmod, train_mse
import numpy as np
import torch
# DeepMoD stuff
import matplotlib.pyplot as plt


# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Loading data
data = np.load('../tests/data/burgers.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten()))
# print("X_size")
# print(X.shape)
# (25856,2) 25856个样本，其中输入t,x
y = np.real(data['u']).reshape((data['u'].size, 1))
# print("y_size")
# print(y.shape)
# (25856,2) 25856个样本，其中u作为groundtruth
number_of_samples = 100

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
# print("X_train_size")
# print(X_train.shape)
# (1000,2) 采样1000个样本
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
# print("y_train_size")
# print(y_train.shape)
# (1000,1) 采样1000个样本

## Running DeepMoD
config = {'n_in': 2, 'hidden_dims': [20, 20, 20, 20, 20, 20], 'n_out': 1, 'library_function': library_1D_in, 'library_args':{'poly_order': 2, 'diff_order': 2}}

model = DeepMod(**config)
optimizer = torch.optim.Adam([{'params': model.network_parameters(), 'lr': 0.002}, {'params': model.coeff_vector(), 'lr':0.002}])
#train_mse(model, X_train, y_train, optimizer, 1000)
train_deepmod(model, X_train, y_train, optimizer, 5000, {'l1': 1e-5})
# iterstot = 5000
# gt = np.array([[ 0],
#         [0],
#         [ 0.1],
#         [0],
#         [-1],
#         [0],
#         [0],
#         [0],
#         [0]])
# rts = {}
# errs = {}
# prim, sp_list = train_deepmod(model, X_train, y_train, optimizer, iterstot, {'l1': 1e-5})
# sp_list = list(map(lambda x : x.cpu().detach().numpy(), sp_list))[0]
# rts[number_of_samples] = prim.cpu().detach().numpy()
# cmpiter = len(sp_list)
# print("cmpiter: ", cmpiter)
# err = 0
# for i in range(cmpiter):
#     err += ((gt[sp_list[i]][0] - rts[number_of_samples][sp_list[i]]))**2
#
# err /= cmpiter
# print("err2: ", err)
# err = err ** 0.5
# errs[number_of_samples] = err
# print(errs)
# plt.plot(list(errs.keys()), list(errs.values()))
# x = list(errs.keys())
# y = list(errs.values())
# for i in range(len(errs)):
#     plt.annotate(str(x[i]) , (x[i], y[i]))
# plt.savefig("./bugers{}.png".format(iterstot))

print()
print()
print(model.fit.sparsity_mask)
print()
print()
print(model.fit.coeff_vector[0])
# print()
# print(model.fit.sparse_theta)
# print(rts)
