# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/19 18:17
@Auth ： Yajun Gao
@File ：7_frozen_filter.py
@IDE ：PyCharm
"""

import torch
import torch.nn as nn
from constrain_moments import K2M

# Create PDENet Block
class PhyCell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))


    def forward(self, x): # x [N, W, H] or [N, 1, W, H]
        x_dims = len(x.shape)
        if x_dims == 3:
            n = x.shape[0]
            w = x.shape[1]
            h = x.shape[2]
            # x [N, W, H] -> x [N, 1, W, H]
            x = x.view(n, 1, w, h)
            x_r = self.F(x)
            next_hidden = x_r.view(n, w, h)

        if x_dims == 4:
            n = x.shape[0]
            w = x.shape[2]
            h = x.shape[3]
            x_r = self.F(x)
            next_hidden = x_r.view(n, 1, w, h)

        return next_hidden

class PDENet():
  def __init__(self, cfg):
      self.cfg = cfg
      self.device = torch.device(self.cfg["device"])
  
  def train(self):
      PhyModel = PhyCell().to(self.device)
      # Moment regularization
      constraints = torch.zeros((49,7,7))
      ind = 0
      for i in range(0,7):
          for j in range(0,7):
              constraints[ind,i,j] = 1
              ind +=1

      phy_loss = 0.0

      k2m = K2M([7, 7])
      for b in range(0, PhyModel.input_dim):
          filters = PhyModel.F.conv1.weight[:, b, :, :] # (nb_filters,7,7)
          m = k2m(filters.double())
          m = m.float()
          # constrains is a precomputed matrix
          phy_loss += nn.MSELoss(m, constraints)
