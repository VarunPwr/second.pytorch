"""The swing leg controller class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import copy
import math
from typing import Any, Tuple

import os
import inspect

from torch.functional import Tensor
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import leg_controller
from torch import Tensor


# The position correction coefficients in Raibert's formula.
_KP = [0.03, 0.03, 0.03]
# At the end of swing, we leave a small clearance to prevent unexpected foot
# collision.
_FOOT_CLEARANCE_M = 0.01


def _gen_parabola(phase: Tensor, start: Tensor, mid: Tensor, end: Tensor) -> Tensor:
    """Gets a point on a parabola y = a x^2 + b x + c.

    The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
    the plane.

    Args:
      phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
      start: The y value at x == 0.
      mid: The y value at x == 0.5.
      end: The y value at x == 1.

    Returns:
      The y value at x == phase.
    """
    mid_phase = 0.5
    delta_1 = mid - start
    delta_2 = end - start
    delta_3 = mid_phase**2 - mid_phase
    coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
    coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
    coef_c = start

    return coef_a * phase**2 + coef_b * phase + coef_c


def _gen_swing_foot_trajectory(input_phase: Tensor, start_pos: Tensor,
                               end_pos: Tensor) -> Tensor:
    """Generates the swing trajectory using a parabola.

    Args:
      input_phase: the swing/stance phase value between [0, 1].
      start_pos: The foot's position at the beginning of swing cycle.
      end_pos: The foot's desired position at the end of swing cycle.

    Returns:
      The desired foot position at the current phase.
    """
    # We augment the swing speed using the below formula. For the first half of
    # the swing cycle, the swing leg moves faster and finishes 80% of the full
    # swing trajectory. The rest 20% of trajectory takes another half swing
    # cycle. Intuitely, we want to move the swing foot quickly to the target
    # landing location and stay above the ground, in this way the control is more
    # robust to perturbations to the body that may cause the swing foot to drop
    # onto the ground earlier than expected. This is a common practice similar
    # to the MIT cheetah and Marc Raibert's original controllers.
    phase = torch.zeros_like(input_phase, device=input_phase.device)
    indices = torch.where(input_phase <= 0.5)
    non_indices = torch.where(input_phase > 0.5)
    phase[indices] = 0.8 * torch.sin(input_phase[indices] * math.pi)
    phase[non_indices] = 0.8 + (input_phase[non_indices] - 0.5) * 0.4

    x = (1 - phase) * start_pos[..., 0] + phase * end_pos[..., 0]
    y = (1 - phase) * start_pos[..., 1] + phase * end_pos[..., 1]
    max_clearance = 0.1
    mid = max(end_pos[..., 2], start_pos[..., 2]) + max_clearance
    z = _gen_parabola(phase, start_pos[..., 2], mid, end_pos[..., 2])

    return torch.stack([x, y, z], dim=-2)


class RaibertSwingLegController(leg_controller.LegController):
    """Controls the swing leg position using Raibert's formula.

    For details, please refer to chapter 2 in "Legged robbots that balance" by
    Marc Raibert. The key idea is to stablize the swing foot's location based on
    the CoM moving speed.

    """

    def __init__(
        self,
        robot_task: Any,
        gait_generator: Any,
        state_estimator: Any,
        desired_speed: Tuple[float, float],
        desired_twisting_speed: float,
        desired_height: float,
        foot_clearance: float,
    ):
        """Initializes the class.

        Args:
          robot: A robot instance.
          gait_generator: Generates the stance/swing pattern.
          state_estimator: Estiamtes the CoM speeds.
          desired_speed: Behavior parameters. X-Y speed.
          desired_twisting_speed: Behavior control parameters.
          desired_height: Desired standing height.
          foot_clearance: The foot clearance on the ground at the end of the swing
            cycle.
        """
        self._robot_task = robot_task
        self._device = self._robot_task.device
        self._num_envs = self._robot_task.num_envs
        self._state_estimator = state_estimator
        self._gait_generator = gait_generator
        self._last_leg_state = gait_generator.desired_leg_state
        self.desired_speed = torch.as_tensor(
            [[desired_speed[0], desired_speed[1], 0]], device=self._device).repeat(self._num_envs, 1)
        self.desired_twisting_speed = desired_twisting_speed
        self._desired_height = torch.as_tensor(
            [[0, 0, desired_height - foot_clearance]], device=self._device).repeat(self._num_envs, 1)

        self._joint_angles = None
        self._phase_switch_foot_local_position = None
        self.reset()

    def reset(self):
        """Called during the start of a swing cycle.

        Args:
          current_time: The wall time in seconds.
        """
        self._last_leg_state = self._gait_generator.desired_leg_state
        self._phase_switch_foot_local_position = self._robot_task._footPositionsInBaseFrame()

    def update(self):
        """Called at each control step.

        Args:
          current_time: The wall time in seconds.
        """
        new_leg_state = self._gait_generator.desired_leg_state

        # Detects phase switch for each leg so we can remember the feet position at
        # the beginning of the swing phase.
        for leg_id, state in enumerate(new_leg_state):
            if (state == gait_generator_lib.LegState.SWING and state != self._last_leg_state[leg_id]):
                self._phase_switch_foot_local_position[leg_id] = (
                    self._robot.GetFootPositionsInBaseFrame()[leg_id])

        self._last_leg_state = copy.deepcopy(new_leg_state)

    def get_action(self):
        com_velocity = copy.deepcopy(
            self._state_estimator.com_velocity_body_frame)
        com_velocity[..., :2] = 0

        _, _, yaw_dot = self._robot_task._getBaseRollPitchYawRate()
        hip_positions = self._robot_task._getHipPositionsInBaseFrame()

        hip_offset = hip_positions
        twisting_vector = torch.as_tensor(
            [[-hip_offset[1], hip_offset[0], 0]], device=self._device).repeat(self._num_envs, 1)
        hip_horizontal_velocity = com_velocity + yaw_dot * twisting_vector
        target_hip_horizontal_velocity = (
            self.desired_speed + self.desired_twisting_speed * twisting_vector)
        foot_target_position = (
            hip_horizontal_velocity *
            self._gait_generator.stance_duration / 2 - torch.as_tensor([_KP], device=self._device).repeat(self._num_envs, 1) *
            (target_hip_horizontal_velocity - hip_horizontal_velocity)
        ) - self._desired_height + torch.as_tensor([[hip_offset[0], hip_offset[1], 0]], device=self._device).repeat(self._num_envs, 1)
        foot_position = _gen_swing_foot_trajectory(
            self._gait_generator.normalized_phase,
            self._phase_switch_foot_local_position, foot_target_position)
        target_leg_indices = torch.where(
            self._gait_generator.desired_leg_state == gait_generator_lib.LegState.SWING)

        return foot_position, target_leg_indices
