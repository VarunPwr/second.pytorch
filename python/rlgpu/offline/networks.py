from typing import Any
import torch
import pytorch_lightning as pl
import torch.nn.functional as F


def variable_hook(grad):
    print("the gradient of C is：", grad)


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
    grad_hook=False,
    dropout=0.2
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
        if grad_hook:
            layers[-2].weight.register_hook(variable_hook)
        if dropout > 0:
            layers += [torch.nn.Dropout(dropout)]

    return torch.nn.Sequential(*layers)


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes,
        output_size,
        grad_hook=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = mlp(
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=output_size,
            grad_hook=grad_hook,
        )

    def forward(self, x):
        return self.layers(x)


NET_DICT = {"mlp": MLP}


class PlNet(pl.LightningModule):
    def __init__(
        self, network_type, input_size, layer_sizes, output_size, dimension_size, grad_hook=False, layer_norm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dimension_size = dimension_size

        self.model = NET_DICT[network_type](
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=output_size,
            dimension_size=dimension_size,
            grad_hook=grad_hook,
            layer_norm=layer_norm,
        )
        # self.model.apply(weights_init)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def step(self, batch, _):
        data = batch[0]
        x, y = data[..., : self.input_size], data[..., self.input_size:]
        logits = self(x)
        policy_loss = F.cross_entropy(
            logits, y.squeeze(1).long())
        greedy_action = torch.argmax(logits, dim=-1)
        acc = (greedy_action == y.squeeze(1)).sum() / x.shape[0]
        return policy_loss, acc

    def training_step(self, batch: Any, batch_idx: int):
        loss, acc = self.step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self.step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        # print("best", best)
        # print("worst", worst)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, acc = self.step(batch, batch_idx)
        self.log("test/loss", loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode="max"),
                "monitor": "val/acc",
            },
        }


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.uniform_(m.weight, -1e-2, 1e-2)
        torch.nn.init.zeros_(m.bias)
