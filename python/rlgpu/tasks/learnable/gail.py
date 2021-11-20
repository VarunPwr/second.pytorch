import torch
import logging
import numpy as np
import json


class GAIL:
    def __init__(self, discriminator, device, optimizer, lr=1e-3,
                 max_grad_norm=0.5, num_disc_updates=1):

        self.discriminator = discriminator
        self.device = device
        self.optimizer = optimizer
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.num_disc_updates = num_disc_updates

    def load_ref_motions(self, ref_motion_names):
        self.motion_names = ref_motion_names
        self._frames = []
        for motion_names in self.motion_names:
            logging.info("Loading motion from: {:s}".format(motion_names))
            with open(motion_names, "r") as f:
                motion_json = json.load(f)
                self._frames.append(motion_json["Frames"])
                logging.info("Loaded motion from {:s}.".format(motion_names))
        self._frames = torch.as_tensor(
            self._frames, dtype=torch.float32, device=self.device)

    def sample_expert(self, batch_size):
        expert_indices = np.random.randint(
            0, len(self._frames), size=batch_size)
        expert_frames = self._frames[expert_indices]
        expert_frames = expert_frames.to(self.device)
        return expert_frames

    def update(self, obs_batch):
        """
        Update discriminator.
        """
        self.discriminator.train()
        self.optimizer.zero_grad()
        batch_size = obs_batch.shape[0]
        expert_frames = self.sample_expert(batch_size)
        expert_logits = self.discriminator(expert_frames)
        expert_loss = torch.sum((expert_logits - 1) ** 2, dim=-1).mean()
        agent_logits = self.discriminator(obs_batch)
        agent_loss = torch.sum((agent_logits + 1) ** 2, dim=-1).mean()

        loss = 0.5 * (expert_loss + agent_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
