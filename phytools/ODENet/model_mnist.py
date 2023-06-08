import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, odeint, tol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.odeint = odeint
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


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


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='phytools/ODENet/dataset', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='phytools/ODENet/dataset', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='phytools/ODENet/dataset', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, device):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class ODENet(object):

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg["adjoint"] == "true":
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
        self.save = self.cfg["save"]
        makedirs(self.save)
        self.logger = get_logger(logpath=os.path.join(self.save, 'logs'), filepath=os.path.abspath(__file__))
        self.device = torch.device(self.cfg["device"])
        self.is_odenet = cfg["network"] == "odenet"
        self.downsampling_method = cfg["downsampling_method"]
        self.data_aug = cfg["data_aug"]
        self.batch_size = cfg["batch_size"]
        self.test_batch_size = cfg["test_batch_size"]
        self.lr = cfg["lr"]
        self.nepochs = cfg["nepochs"]
        self.tol = cfg["tol"]

        if self.downsampling_method == 'conv':
            downsampling_layers = [
                nn.Conv2d(1, 64, 3, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
            ]
        elif self.downsampling_method == 'res':
            downsampling_layers = [
                nn.Conv2d(1, 64, 3, 1),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ]

        self.feature_layers = [ODEBlock(ODEfunc(64), odeint, self.tol)] if self.is_odenet else [ResBlock(64, 64) for _ in range(6)]
        fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

        self.model = nn.Sequential(*downsampling_layers, *self.feature_layers, *fc_layers).to(self.device)

        self.logger.info(self.model)
        self.logger.info('Number of parameters: {}'.format(count_parameters(self.model)))

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.train_loader, self.test_loader, self.train_eval_loader = get_mnist_loaders(
            self.data_aug, self.batch_size, self.test_batch_size
        )

        self.data_gen = inf_generator(self.train_loader)
        self.batches_per_epoch = len(self.train_loader)

        self.lr_fn = self.learning_rate_with_decay(
            self.batch_size, batch_denom=128, batches_per_epoch=self.batches_per_epoch, boundary_epochs=[60, 100, 140],
            decay_rates=[1, 0.1, 0.01, 0.001]
        )

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        self.batch_time_meter = RunningAverageMeter()
        self.f_nfe_meter = RunningAverageMeter()
        self.b_nfe_meter = RunningAverageMeter()
        self.end = time.time()

    def train(self):
        best_acc = 0
        for itr in range(self.nepochs * self.batches_per_epoch):

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_fn(itr)

            self.optimizer.zero_grad()
            x, y = self.data_gen.__next__()
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            if self.is_odenet:
                nfe_forward = self.feature_layers[0].nfe
                self.feature_layers[0].nfe = 0

            loss.backward()
            self.optimizer.step()

            if self.is_odenet:
                nfe_backward = self.feature_layers[0].nfe
                self.feature_layers[0].nfe = 0

            self.batch_time_meter.update(time.time() - self.end)
            if self.is_odenet:
                self.f_nfe_meter.update(nfe_forward)
                self.b_nfe_meter.update(nfe_backward)
            self.end = time.time()

            if itr % self.batches_per_epoch == 0:
                with torch.no_grad():
                    train_acc = accuracy(self.model, self.train_eval_loader, self.device)
                    val_acc = accuracy(self.model, self.test_loader, self.device)
                    if val_acc > best_acc:
                        torch.save({'state_dict': self.model.state_dict()},
                                   os.path.join(self.save, 'model.pth'))
                        best_acc = val_acc
                    self.logger.info(
                        "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                        "Train Acc {:.4f} | Test Acc {:.4f}".format(
                            itr // self.batches_per_epoch, self.batch_time_meter.val, self.batch_time_meter.avg, self.f_nfe_meter.avg,
                            self.b_nfe_meter.avg, train_acc, val_acc
                        )
                    )

    def learning_rate_with_decay(self, batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
        initial_learning_rate = self.lr * batch_size / batch_denom

        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        vals = [initial_learning_rate * decay for decay in decay_rates]

        def learning_rate_fn(itr):
            lt = [itr < b for b in boundaries] + [True]
            i = np.argmax(lt)
            return vals[i]

        return learning_rate_fn
