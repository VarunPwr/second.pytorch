import math
import torch
import torch.nn as nn
import numpy as np
from rl_pytorch.ppo import init


def get_encoder(encoder_type):
    if encoder_type == "mlp":
        return None
    elif encoder_type == "nature":
        return NatureFuseEncoder
    else:
        raise NotImplementedError


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class MLPBase(nn.Module):
    def __init__(
            self,
            input_shape,
            hidden_shapes,
            activation_func=nn.ReLU,
            init_func=init.basic_init,
            add_ln=False,
            last_activation_func=None):
        super().__init__()

        self.activation_func = activation_func
        self.fcs = []
        self.add_ln = add_ln
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for next_shape in hidden_shapes:
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            self.fcs.append(activation_func())
            if self.add_ln:
                self.fcs.append(nn.LayerNorm(next_shape))
            input_shape = next_shape
            self.output_shape = next_shape

        self.fcs.pop(-1)
        self.fcs.append(self.last_activation_func())
        self.seq_fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        return self.seq_fcs(x)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim, proj=True):
        super().__init__()
        self.out_dim = out_dim
        module_list = [
            nn.Linear(in_dim, out_dim)
        ]
        if proj:
            module_list += [
                nn.ReLU()
            ]

        self.projection = nn.Sequential(
            *module_list
        )
        self.output_dim = out_dim
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class NatureEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 groups=1,
                 flatten=True,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureEncoder, self).__init__()
        self.groups = groups
        layer_list = [
            nn.Conv2d(in_channels=in_channels, out_channels=32 * self.groups,
                      kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.groups, out_channels=64 * self.groups,
                      kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64 * self.groups, out_channels=64 * self.groups,
                      kernel_size=3, stride=1), nn.ReLU(),
        ]
        if flatten:
            layer_list.append(
                nn.Flatten()
            )
        self.layers = nn.Sequential(*layer_list)

        self.output_dim = 1024 * self.groups
        self.apply(orthogonal_init)

    def forward(self, x):
        x = x.view(torch.Size(
            [np.prod(x.size()[:-3])]) + x.size()[-3:])
        x = self.layers(x)
        return x


class NatureFuseEncoder(nn.Module):
    def __init__(self,
                 cfg,
                 **kwargs):
        super(NatureFuseEncoder, self).__init__()
        self.visual_base = NatureEncoder(
            cfg["in_channels"]
        )

        self.in_channels = cfg["in_channels"]
        self.visual_dim = cfg["visual_dim"]
        self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
        self.state_dim = cfg["state_dim"]
        self.visual_projector = RLProjection(
            in_dim=self.visual_base.output_dim,
            out_dim=cfg["hidden_dims"][-1]
        )

        self.base = MLPBase(
            input_shape=cfg["state_dim"],
            hidden_shapes=cfg["hidden_dims"],
            **kwargs
        )

        self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

    def forward(self, x):
        state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
        visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
        visual_out = self.visual_base(visual_x)
        visual_out = self.visual_projector(visual_out)
        state_out = self.base(state_x)

        return torch.cat([visual_out, state_out], dim=-1)
