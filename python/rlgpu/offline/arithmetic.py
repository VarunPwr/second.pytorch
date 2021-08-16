import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.parameter import Parameter

EPS = 1e-8


class NeuralAddUnitCell(nn.Module):

    def __init__(self, in_dim, out_dim, reset=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('bias', None)
        if reset:
            self._reset_params()

    def _reset_params(self):
        std = math.sqrt(2.0 / (self.in_dim + self.out_dim))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W_hat, -r, r)

    def forward(self, x):
        W = torch.clamp(self.W_hat, -1, 1)
        return F.linear(x, W, self.bias)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

    def regularizer(self):
        return torch.mean(torch.min(torch.abs(self.W_hat), 1-torch.abs(self.W_hat)))


class NAU(nn.Module):

    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        sizes = [input_size] + layer_sizes + [output_size]

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(
                NeuralAddUnitCell(
                    sizes[i], sizes[i + 1]
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class NeuralPowerUnitCell(nn.Module):

    def __init__(self, in_dim, out_dim, reset=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.W_i = Parameter(torch.Tensor(out_dim, in_dim))

        self.gate = Parameter(torch.ones(in_dim, ))
        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('W_i', self.W_i)
        self.register_parameter('gate', self.gate)
        self.register_parameter('bias', None)
        if reset:
            self._reset_params()

    def _reset_params(self):
        torch.nn.init.xavier_uniform_(self.W_hat)
        torch.nn.init.xavier_uniform_(self.W_i)

    def forward(self, x):
        g = torch.clamp(self.gate, 0, 1)
        r = torch.abs(x) + EPS
        r = g * r + (1 - g)
        k = torch.clamp(-torch.sign(x), min=0) * g * math.pi
        z = torch.exp(F.linear(torch.log(r), self.W_hat, self.bias) - F.linear(k, self.W_i, self.bias)) * \
            torch.cos(F.linear(k, self.W_hat, self.bias) +
                      F.linear(torch.log(r), self.W_i, self.bias))
        return z

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

    def regularizer(self):
        return torch.mean(torch.min(self.W_hat, 1 - self.W_hat))


class ArithmeticNet(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, grad_hook=False,):
        super().__init__()
        del grad_hook
        self.layers = [
            NAU(input_size, layer_sizes, input_size),
            NeuralPowerUnitCell(input_size, output_size),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
