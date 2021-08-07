"""State estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any

from rlgpu.utils.torch_jit_utils import quat_rotate_inverse
import torch
from torch.tensor import Tensor
from isaacgym import gymapi
from pytorch3d.transforms import quaternion_apply

_DEFAULT_WINDOW_SIZE = 20


class MovingWindowFilter(object):
    """A stable O(1) moving filter for incoming data streams.

    We implement the Neumaier's algorithm to calculate the moving window average,
    which is numerically stable.

    """

    def __init__(self, robot_task: Any, window_size: int):
        """Initializes the class.

        Args:
          window_size: The moving window size.
        """
        assert window_size > 0
        self._robot_task = robot_task
        self._num_envs = self._robot_task.num_envs
        self._window_size = window_size
        self._current_size = 0
        self._device = self._robot_task.device
        self._value_deque = torch.zeros(
            (self._num_envs, self._window_size), device=self._device)
        # The moving window sum.
        self._sum = torch.zeros((self._num_envs), device=self._device)
        # The correction term to compensate numerical precision loss during
        # calculation.
        self._correction = torch.zeros((self._num_envs), device=self._device)

    def _neumaier_sum(self, value: Tensor) -> Tensor:
        """Update the moving window sum using Neumaier's algorithm.

        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

        Args:
          value: The new value to be added to the window.
        """

        new_sum = self._sum + value
        bigger_indices = torch.stack(torch.nonzero(torch.abs(self._sum) >= torch.abs(value), as_tuple=True))
        smaller_indices = torch.stack(torch.nonzero(torch.abs(self._sum) < torch.abs(value), as_tuple=True))
        self._correction[bigger_indices] += ((self._sum - new_sum) + value)[bigger_indices]
        self._correction[smaller_indices] += ((value - new_sum) + self._sum)[smaller_indices]
        self._sum = new_sum

    def calculate_average(self, new_value: Tensor) -> Tensor:
        """Computes the moving window average in O(1) time.

        Args:
          new_value: The new value to enter the moving window.

        Returns:
          The average of the values in the window.

        """
        if self._current_size < self._window_size:
            pass
        else:
            # The left most value to be subtracted from the moving sum.
            self._neumaier_sum(-self._value_deque[..., 0])

        self._neumaier_sum(new_value)

        self.append_deque(new_value)

        return (self._sum + self._correction) / self._window_size

    def append_deque(self, new_value: Tensor):
        if self._current_size < self._window_size:
            self._value_deque[..., self._current_size] = new_value
            self._current_size += 1
        else:
            self._value_deque = torch.cat(
                [self._value_deque[..., :-1], new_value], dim=-1)


class COMVelocityEstimator(object):
    """Estimate the CoM velocity using on board sensors.


    Requires knowledge about the base velocity in world frame, which for example
    can be obtained from a MoCap system. This estimator will filter out the high
    frequency noises in the velocity so the results can be used with controllers
    reliably.

    """

    def __init__(
        self,
        robot_task: Any,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ):
        self._robot_task = robot_task
        self._num_envs = self._robot_task.num_envs
        self._device = self._robot_task.device
        self._window_size = window_size
        self.reset()

    @property
    def com_velocity_body_frame(self) -> Tensor:
        """The base velocity projected in the body aligned inertial frame.

        The body aligned frame is a intertia frame that coincides with the body
        frame, but has a zero relative velocity/angular velocity to the world frame.

        Returns:
          The com velocity in body aligned frame.
        """
        return self._com_velocity_body_frame

    @property
    def com_velocity_world_frame(self) -> Tensor:
        return self._com_velocity_world_frame

    def reset(self):
        # We use a moving window filter to reduce the noise in velocity estimation.
        self._velocity_filter_x = MovingWindowFilter(
            robot_task=self._robot_task, window_size=self._window_size)
        self._velocity_filter_y = MovingWindowFilter(
            robot_task=self._robot_task, window_size=self._window_size)
        self._velocity_filter_z = MovingWindowFilter(
            robot_task=self._robot_task, window_size=self._window_size)
        self._com_velocity_world_frame = torch.zeros(
            (self._num_envs, 3), device=self._device)
        self._com_velocity_body_frame = torch.zeros(
            (self._num_envs, 3), device=self._device)

    def update(self):
        robot_states = self._robot_task.root_states[self._robot_task.a1_indices]
        base_quat = robot_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(
            base_quat, robot_states[:, 7:10])
        vx = self._velocity_filter_x.calculate_average(base_lin_vel[..., 0])
        vy = self._velocity_filter_y.calculate_average(base_lin_vel[..., 1])
        vz = self._velocity_filter_z.calculate_average(base_lin_vel[..., 2])
        self._com_velocity_world_frame = torch.stack([vx, vy, vz], dim=-1)
        self._com_velocity_body_frame = quaternion_apply(base_quat, self._com_velocity_world_frame)
