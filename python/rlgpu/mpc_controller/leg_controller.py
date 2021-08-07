"""The leg controller class interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any
import torch

class LegController(object):
    """Generates the leg control signal."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self, current_time: float):
        """Resets the controller's internal state."""
        pass

    @abc.abstractmethod
    def update(self, current_time: float):
        """Updates the controller's internal state."""
        pass

    @abc.abstractmethod
    def get_action(self) -> Any:
        """Gets the control signal e.g. torques/positions for the leg."""
        pass

    def set_desired_speed_as_tensor(self, speed):
        self.desired_speed = torch.as_tensor(speed, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)

    def set_desired_twisting_speed_as_tensor(self, speed):
        self.desired_twisting_speed = torch.as_tensor(speed, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)