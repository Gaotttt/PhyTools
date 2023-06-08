import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


class Lambda(nn.Module):

    def __init__(self, device):
        super(Lambda, self).__init__()
        self.true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

    def forward(self, t, y):
        return torch.mm(y ** 3, self.true_A)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y ** 3)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class ODENet(object):

    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg["adjoint"] == "true":
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.device = torch.device(self.cfg["device"])

        self.true_y0 = torch.tensor([[2., 0.]]).to(self.device)
        self.t = torch.linspace(0., 25., self.cfg["data_size"]).to(self.device)
        self.true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(self.device)

        with torch.no_grad():
            self.true_y = odeint(Lambda(self.device), self.true_y0, self.t, method='dopri5')

        if self.cfg["viz"] == 'true':
            self.makedirs('png')

            self.fig = plt.figure(figsize=(12, 4), facecolor='white')
            self.ax_traj = self.fig.add_subplot(131, frameon=False)
            self.ax_phase = self.fig.add_subplot(132, frameon=False)
            self.ax_vecfield = self.fig.add_subplot(133, frameon=False)
            plt.show(block=False)

    def train(self):

        if self.cfg["adjoint"] == "true":
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        ii = 0

        func = ODEFunc().to(self.device)

        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        end = time.time()

        time_meter = RunningAverageMeter(0.97)

        loss_meter = RunningAverageMeter(0.97)

        for itr in range(1, self.cfg["niters"] + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = self.get_batch()
            pred_y = odeint(func, batch_y0, batch_t).to(self.device)
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

            if itr % self.cfg["test_freq"] == 0:
                with torch.no_grad():
                    pred_y = odeint(func, self.true_y0, self.t)
                    loss = torch.mean(torch.abs(pred_y - self.true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    self.visualize(self.true_y, pred_y, func, ii)
                    ii += 1

            end = time.time()

    def get_batch(self):
        s = torch.from_numpy(
            np.random.choice(np.arange(self.cfg["data_size"] - self.cfg["batch_time"], dtype=np.int64), self.cfg["batch_size"],
                             replace=False))
        batch_y0 = self.true_y[s]  # (M, D)
        batch_t = self.t[:self.cfg["batch_time"]]  # (T)
        batch_y = torch.stack([self.true_y[s + i] for i in range(self.cfg["batch_time"])], dim=0)  # (T, M, D)
        return batch_y0.to(self.device), batch_t.to(self.device), batch_y.to(self.device)

    def makedirs(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def visualize(self, true_y, pred_y, odefunc, itr):
        if self.cfg["viz"]:
            self.ax_traj.cla()
            self.ax_traj.set_title('Trajectories')
            self.ax_traj.set_xlabel('t')
            self.ax_traj.set_ylabel('x,y')
            self.ax_traj.plot(self.t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], self.t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                         'g-')
            self.ax_traj.plot(self.t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', self.t.cpu().numpy(),
                         pred_y.cpu().numpy()[:, 0, 1], 'b--')
            self.ax_traj.set_xlim(self.t.cpu().min(), self.t.cpu().max())
            self.ax_traj.set_ylim(-2, 2)
            self.ax_traj.legend()

            self.ax_phase.cla()
            self.ax_phase.set_title('Phase Portrait')
            self.ax_phase.set_xlabel('x')
            self.ax_phase.set_ylabel('y')
            self.ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
            self.ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
            self.ax_phase.set_xlim(-2, 2)
            self.ax_phase.set_ylim(-2, 2)

            self.ax_vecfield.cla()
            self.ax_vecfield.set_title('Learned Vector Field')
            self.ax_vecfield.set_xlabel('x')
            self.ax_vecfield.set_ylabel('y')

            y, x = np.mgrid[-2:2:21j, -2:2:21j]
            dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(self.device)).cpu().detach().numpy()
            mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
            dydt = (dydt / mag)
            dydt = dydt.reshape(21, 21, 2)

            self.ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
            self.ax_vecfield.set_xlim(-2, 2)
            self.ax_vecfield.set_ylim(-2, 2)

            self.fig.tight_layout()
            plt.savefig('png/{:03d}'.format(itr))
            plt.draw()
            plt.pause(0.001)
