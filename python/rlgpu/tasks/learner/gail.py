import torch
from rl_pytorch.networks import get_network
import logging
import numpy as np
import json
import os

from torch.optim.adam import Adam


class GAIL:
    def __init__(self, num_envs, device, learner_cfg):

        self.discriminator = get_network("discriminator", learner_cfg)
        self.discriminator.to(device)
        self.num_envs = num_envs
        self.nsteps = learner_cfg["nsteps"]
        self.nminibatches = learner_cfg["nminibatches"] * self.num_envs
        self.device = device
        self.optimizer = Adam(
            self.discriminator.parameters(), lr=learner_cfg["lr"])
        self.max_grad_norm = learner_cfg["max_grad_norm"]
        self.noptepochs = learner_cfg["noptepochs"]
        self.dof = learner_cfg["dof"]
        self.transtion_buffer = torch.zeros(
            (self.nsteps, self.num_envs, 2 * self.dof)).to(device)  # 2 * 2 * dof means we have dof pos + dof vel, for current and next pose
        self.step = 0
        self.motion_names = learner_cfg["motion_names"]
        self.load_ref_motions()

    def load_ref_motions(self):
        self._frames = []
        for motion_names in self.motion_names:
            motion_file_names = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), '../../data/a1_motions', motion_names + ".txt")
            logging.info("Loading motion from: {:s}".format(motion_names))
            with open(motion_file_names, "r") as f:
                motion_json = json.load(f)
                self._frames.append(motion_json["Frames"])
                logging.info("Loaded motion from {:s}.".format(motion_names))
        self._frames = torch.as_tensor(
            self._frames, dtype=torch.float32, device=self.device)[..., -self.dof:]
        self._current_frames = self._frames[:, :-1]
        self._next_frames = self._frames[:, 1:]
        self._frames = torch.cat(
            [self._current_frames, self._next_frames], dim=-1).view(-1, self.dof * 2)

    def sample_expert(self, nminibatches):
        expert_indices = np.random.randint(
            0, len(self._frames), size=nminibatches)
        expert_frames = self._frames[expert_indices]
        expert_frames = expert_frames.to(self.device)
        return expert_frames

    def sample_transitions(self, nminibatches):
        """
        Sample transitions.
        """
        indices = np.random.randint(
            0, self.nsteps, size=nminibatches)
        return self.transtion_buffer[indices]

    def save_transition(self, current_pose, next_pose):
        """
        Save transition.
        """
        assert self.step <= self.nsteps, "GAIL transition buffer overflow!"

        transition = torch.cat((current_pose, next_pose), dim=-1)
        self.transtion_buffer[self.step].copy_(transition)
        self.step += 1

    def clear(self):
        self.step = 0

    def check_update(self):
        """
        Check if we need to update the discriminator.
        """
        return self.step >= self.nsteps

    def update(self):
        """
        Update discriminator.
        """
        self.discriminator.train()
        total_loss = 0
        for _ in range(self.noptepochs):
            expert_frames = self.sample_expert(self.nminibatches)
            agent_frames = self.sample_transitions(self.nminibatches)
            for expert_frame, agent_frame in zip(expert_frames, agent_frames):
                expert_frame = expert_frame.view(
                    self.num_envs, -1)
                agent_frame = agent_frame.view(
                    self.num_envs, -1)
                self.optimizer.zero_grad()
                expert_logits = self.discriminator(expert_frame)
                expert_loss = torch.sum(
                    (expert_logits - 1) ** 2, dim=-1).mean()
                agent_logits = self.discriminator(agent_frame)
                agent_loss = torch.sum((agent_logits + 1) ** 2, dim=-1).mean()
                loss = 0.5 * (expert_loss + agent_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                               self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
        self.clear()
        mean_loss = total_loss * self.num_envs / self.noptepochs / self.nminibatches
        print("#" * 25, "GAIL update", "#" * 25)
        print(" " * 20, "Mean loss: {:.4f}".format(mean_loss))
        print()
        return mean_loss

    def reward(self, current_pose: torch.Tensor, next_pose: torch.Tensor):
        """
        Compute reward.
        """
        with torch.no_grad():
            transition = torch.cat((current_pose, next_pose), dim=-1)
            pred_logits = self.discriminator(transition).squeeze(-1)
            pred_scores = 1 - 0.25 * (pred_logits - 1) ** 2
            reward = torch.clamp(pred_scores, 0.0, 1.0)
            return reward
