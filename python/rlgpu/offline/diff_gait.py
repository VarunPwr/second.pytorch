import torch
import torch.nn as nn
import torch.nn.functional as F


CONTACT_FORCE_SCALE = [[0, 0, 0], [0, 0, -10], [0, 0, 0]]
class DifferentiableGaitLib(nn.Module):
    def __init__(self, num_gaits=4, num_legs=4, dim=3, init_scale=CONTACT_FORCE_SCALE):
        super().__init__()
        self.num_gaits = num_gaits
        self.num_legs = num_legs
        self.dim = dim
        self.params = nn.Parameter(torch.Tensor(num_gaits, dim).uniform_(-1, 1))
        if init_scale is not None:
            init_scale = torch.as_tensor(init_scale)
            assert init_scale.shape == self.params.shape

    def forward(self, logits):
        assert logits.shape[-1] == self.num_gaits and logits.shape[-2] == self.num_legs
        hard_prob = F.gumbel_softmax(logits, dim=-1, hard=True)
        gaits = self.params.unsqueeze(0).unsqueeze(0).repeat(logits.shape[0], logits.shape[1], 1, 1) * hard_prob.unsqueeze(-1)
        gaits = gaits.sum(-2)
        return gaits

