from typing import Any
from qpth.qp import QPFunction
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl
import torch.nn.functional as F
from siren_pytorch import SirenNet
from diff_gait import DifferentiableGaitLib
from arithmetic import ArithmeticNet


def variable_hook(grad):
    print("the gradient of C is：", grad)


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.Softsign,
    grad_hook=False,
    dropout=0.0,
    ln=False
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    if ln:
        layers.append(nn.LayerNorm(input_size))
    # layers.append(nn.LayerNorm(input_size))
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
        if ln:
            layers.append(torch.nn.LayerNorm(sizes[i + 1]))
        if grad_hook:
            layers[-2].weight.register_hook(variable_hook)
        if dropout > 0 and i < len(sizes) - 2:
            layers += [torch.nn.Dropout(dropout)]
    return torch.nn.Sequential(*layers)


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes,
        output_size,
        grad_hook=False,
        ln=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = mlp(
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=output_size,
            grad_hook=grad_hook,
            ln=ln
        )

    def forward(self, x):
        return self.layers(x)


class OptNet(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, nineq=200, neq=0, eps=1e-4, grad_hook=False, ln=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        self.fc = mlp(input_size, layer_sizes,
                      output_size, grad_hook=grad_hook)

        self.M = nn.Parameter(torch.tril(torch.ones(output_size, output_size)))
        self.L = nn.Parameter(torch.tril(torch.rand(output_size, output_size)))
        self.G = nn.Parameter(torch.Tensor(
            nineq, output_size).uniform_(-1e-2, 1e-2))
        self.z0 = nn.Parameter(torch.zeros(output_size))
        self.s0 = nn.Parameter(torch.ones(nineq))

    def forward(self, x):
        x = self.fc(x)
        L = self.M * self.L
        Q = L.mm(L.t()) + self.eps * \
            Variable(torch.eye(self.output_size)).to(x.device)
        h = self.G.mv(self.z0) + self.s0
        e = Variable(torch.Tensor()).to(x.device)
        x = QPFunction(verbose=False)(Q, x, self.G, h, e, e)

        return x


class GaitNet(nn.Module):
    def __init__(self, input_size,
                 layer_sizes,
                 output_size,
                 grad_hook=False, num_gaits=3, num_legs=4, ln=False):
        super().__init__()
        self.num_gaits = num_gaits
        self.num_legs = num_legs
        self.gait_lib_dim = output_size // self.num_legs
        self.lib = DifferentiableGaitLib(
            num_gaits=self.num_gaits, num_legs=num_legs, dim=self.gait_lib_dim)
        self.fc = mlp(input_size, layer_sizes, self.num_gaits *
                      self.num_legs, grad_hook=grad_hook)
        self.residual = mlp(input_size, layer_sizes,
                            output_size, grad_hook=grad_hook)
        self.scaling = nn.Linear(output_size, output_size)

    def forward(self, x):
        logits = self.fc(x).view(-1, self.num_legs, self.num_gaits)
        output = self.lib(logits)
        # return output.flatten(1) + residual
        return self.scaling(output.flatten(1)) + self.residual(x)


NET_DICT = {"mlp": MLP, "optnet": OptNet, "siren": SirenNet,
            "gaitnet": GaitNet, "arithmetic": ArithmeticNet}


class PlNet(pl.LightningModule):
    def __init__(
            self, network_type, input_size, layer_sizes, output_size, grad_hook=False, ln=False, tanh_loss=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        if network_type == "siren":
            self.model = NET_DICT[network_type](
                dim_in=input_size,
                dim_hidden=layer_sizes[0],
                dim_out=output_size,
                num_layers=len(layer_sizes),
                final_activation=nn.Identity(),
                w0_initial=1.
            )
        else:
            self.model = NET_DICT[network_type](
                input_size=input_size,
                layer_sizes=layer_sizes,
                output_size=output_size,
                grad_hook=grad_hook,
                ln=ln
            )
        self.tanh_loss = tanh_loss
        # self.model.apply(weights_init)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def step(self, batch, _):
        data = batch[0].float()
        x, y = data[..., : self.input_size], data[..., self.input_size:]
        logits = self(x)

        relative_error = (torch.abs(y - logits) /
                          (1e-4 + torch.abs(y))).mean().detach()

        if self.tanh_loss:
            logits = torch.tanh(logits)
            y = torch.tanh(y)
        policy_loss = F.mse_loss(logits, y)
        return policy_loss, relative_error

    def training_step(self, batch: Any, batch_idx: int):
        loss, err = self.step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, sync_dist=True)

        self.log("train/err", err, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, err = self.step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/err", err, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, err = self.step(batch, batch_idx)
        self.log("test/loss", loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log("test/err", err, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode="min"),
                "monitor": "val/loss",
            },
        }


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.uniform_(m.weight, -1e-2, 1e-2)
        torch.nn.init.zeros_(m.bias)
