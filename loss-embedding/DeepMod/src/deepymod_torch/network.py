import torch
import torch.nn as nn


class Library(nn.Module):
    def __init__(self, library_func, library_args={}):
        super().__init__()
        self.library_func = library_func
        self.library_args = library_args

    def forward(self, input):
        time_deriv_list, theta = self.library_func(input, **self.library_args)
        return time_deriv_list, theta


class Fitting(nn.Module):
    def __init__(self, n_terms, n_out):
        super().__init__()
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(torch.rand((n_terms, 1), dtype=torch.float32)) for _ in torch.arange(n_out)])
        # 这里创建了一个n_out长度的list，每个list是一个n_terms*1的tensor[0--->n_terms-1]
        self.sparsity_mask = [torch.arange(n_terms) for _ in torch.arange(n_out)]


    def forward(self, input):
        # 这里的input应该是theta
        # print("input")
        # print(input.shape)
        # (1000,9)
        sparse_theta = self.apply_mask(input)
        return sparse_theta, self.coeff_vector

    def apply_mask(self, theta):
        # print("theta")
        # print(theta.shape)
        # (1000,9)
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        # print("sparse_theta")
        # print(len(sparse_theta))
        # # 1
        # print(sparse_theta[0].shape)
        # # (1000,9)
        return sparse_theta
