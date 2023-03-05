import sys
import warnings
from os.path import join as ospj
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
sys.path.insert(1, '../..')
from src.deepymod_torch.network import Fitting, Library
import torch
import torch.nn as nn
# import sys
# sys.path.append("..")



class DeepMod(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, n_in, hidden_dims, n_out, library_function, library_args):
        super().__init__()
        self.network = self.build_network(n_in, hidden_dims, n_out)
        self.library = Library(library_function, library_args)
        self.fit = self.build_fit_layer(n_in, n_out, library_function, library_args)

    def forward(self, input):
        prediction = self.network(input)
        # print("input")
        # print(input.shape)
        # # ks (5000,2)
        # print("prediction")
        # print(prediction.shape)
        # # ks (5000,2)
        time_deriv, theta = self.library((prediction, input))
        # print("theta")
        # print(theta.shape)
        # print(len(time_deriv))
        # print(time_deriv[0].shape)
        # # list 2 (5000,1)
        # (1000,9)
        # print("fit come on")
        sparse_theta, coeff_vector = self.fit(theta)
        # print("fit end")
        # print("sparse_theta")
        # print(len(sparse_theta))
        # print(sparse_theta[0].shape)
        # # 1 list (1000,9)
        # print("coeff_vector")
        # print(coeff_vector[0].shape)
        return prediction, time_deriv, sparse_theta, coeff_vector

    def build_network(self, n_in, hidden_dims, n_out):
        # NN
        network = []
        hs = [n_in] + hidden_dims + [n_out]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 

        return network

    # 这里怎么实现的 fit，以及为何要实现fit
    def build_fit_layer(self, n_in, n_out, library_function, library_args):
        # print("build fit layer")
        # print("n_in")
        # print(n_in)
        # print("n_out")
        # print(n_out)
        sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
        # print("sample_input")
        # print(sample_input.shape)
        # do sample pass to infer shapes
        n_terms = self.library((self.network(sample_input), sample_input))[1].shape[1]
        # print("n_terms")
        # print(n_terms)
        fit_layer = Fitting(n_terms, n_out)
        # print("build fit layer end")
        return fit_layer

    # Function below make life easier
    def network_parameters(self):
        return self.network.parameters()

    def coeff_vector(self):
        return self.fit.coeff_vector.parameters()
        # return self.fit.coeff_vector


# class DeepMod(nn.Module):
#     ''' Class based interface for deepmod.'''
#     def __init__(self, n_in, hidden_dims, n_out, library_function, library_args):
#         super().__init__()
#         self.network = self.build_network(n_in, hidden_dims, n_out)
#         self.library = Library(library_function, library_args)
#         self.fit = self.build_fit_layer(n_in, n_out, library_function, library_args)
#
#     def forward(self, input):
#         prediction = self.network(input)
#         # print("input")
#         # print(input.shape)
#         # # ks (5000,2)
#         # print("prediction")
#         # print(prediction.shape)
#         # # ks (5000,2)
#         time_deriv, theta = self.library((prediction, input))
#         # print("theta")
#         # print(theta.shape)
#         # print(len(time_deriv))
#         # print(time_deriv[0].shape)
#         # # list 2 (5000,1)
#         # (1000,9)
#         # print("fit come on")
#         sparse_theta, coeff_vector = self.fit(theta)
#         # print("fit end")
#         # print("sparse_theta")
#         # print(len(sparse_theta))
#         # print(sparse_theta[0].shape)
#         # # 1 list (1000,9)
#         # print("coeff_vector")
#         # print(coeff_vector[0].shape)
#         return prediction, time_deriv, sparse_theta, coeff_vector
#
#     def build_network(self, n_in, hidden_dims, n_out):
#         # NN
#         network = []
#         hs = [n_in] + hidden_dims + [n_out]
#         for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
#             network.append(nn.Linear(h0, h1))
#             network.append(nn.Tanh())
#         network.pop()  # get rid of last activation function
#         network = nn.Sequential(*network)
#
#         return network
#
#     # 这里怎么实现的 fit，以及为何要实现fit
#     def build_fit_layer(self, n_in, n_out, library_function, library_args):
#         # print("build fit layer")
#         # print("n_in")
#         # print(n_in)
#         # print("n_out")
#         # print(n_out)
#         sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
#         # print("sample_input")
#         # print(sample_input.shape)
#         # do sample pass to infer shapes
#         n_terms = self.library((self.network(sample_input), sample_input))[1].shape[1]
#         # print("n_terms")
#         # print(n_terms)
#         fit_layer = Fitting(n_terms, n_out)
#         # print("build fit layer end")
#         return fit_layer
#
#     # Function below make life easier
#     def network_parameters(self):
#         return self.network.parameters()
#
#     def coeff_vector(self):
#         return self.fit.coeff_vector.parameters()
#         # return self.fit.coeff_vector
