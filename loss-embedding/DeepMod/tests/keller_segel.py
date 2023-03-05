# Keller Segel, tests coupled output.

# General imports
import numpy as np
import torch
# DeepMoD stuff
import sys
import warnings
from os.path import join as ospj
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
sys.path.insert(1, '../..')
from src.deepymod_torch.DeepMod import DeepMod
from src.deepymod_torch.library_functions import library_1D_in
from src.deepymod_torch.training import train_deepmod, train_mse
from src.deepymod_torch.utilities import create_deriv_data

# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Loading data
data = np.load('../tests/data/keller_segel.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten()))
# (10201, 2)
y = np.transpose((data['u'].flatten(), data['v'].flatten()))
# (10201, 2)
number_of_samples = 5000

# 对10201之间随机排序，实现随机取5000个数据
idx = np.random.permutation(y.shape[0])
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
# torch.Size([5000, 2])
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
# torch.Size([5000, 2])

## Running DeepMoD
# config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 2, 'library_function': library_1D_in, 'library_args':{'poly_order': 1, 'diff_order': 2}}
config = {'n_in': 2, 'hidden_dims': [20, 20, 20, 20, 20, 20], 'n_out': 2, 'library_function': library_1D_in, 'library_args':{'poly_order': 1, 'diff_order': 2}}
model = DeepMod(**config)

# X_input = create_deriv_data(X_train, config['library_args']['diff_order'])
# torch.Size([5000, 2, 2, 2])
optimizer = torch.optim.Adam([{'params': model.network_parameters(), 'lr': 0.002}, {'params': model.coeff_vector(), 'lr':0.002}])
# optimizer = torch.optim.Adam(model.parameters())
train_deepmod(model, X_train, y_train, optimizer, 25000, {'l1': 1e-5})

print()
print()
print(model.fit.sparsity_mask)
print()
print()
print(model.fit.coeff_vector)
print(model.fit.coeff_vector[0])
print(model.fit.coeff_vector[1])
