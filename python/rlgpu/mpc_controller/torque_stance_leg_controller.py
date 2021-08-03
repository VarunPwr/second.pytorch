# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Sequence, Tuple
from torch import Tensor
from pytorch3d.transforms import quaternion_to_matrix
import numpy as np
import torch
from copy import deepcopy

# import time

from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import leg_controller
from mpc_controller import qp_torque_optimizer

_FORCE_DIMENSION = 3
KP = np.array((0., 0., 100., 100., 100., 0.))
KD = np.array((40., 30., 10., 10., 10., 30.))
MAX_DDQ = np.array((10., 10., 10., 20., 20., 20.))
MIN_DDQ = -MAX_DDQ


class TorqueStanceLegController(leg_controller.LegController):
    """A torque based stance leg controller framework.

    Takes in high level parameters like walking speed and turning speed, and
    generates necessary the torques for stance legs.
    """

    def __init__(
        self,
        robot_task: Any,
        gait_generator: Any,
        state_estimator: Any,
        desired_speed: Tuple[float, float] = (0, 0),
        desired_twisting_speed: float = 0,
        desired_body_height: float = 0.45,
        friction_coeffs: Sequence[float] = (0.45, 0.45, 0.45, 0.45),
    ):
        """Initializes the class.

        Tracks the desired position/velocity of the robot by computing proper joint
        torques using MPC module.

        Args:
          robot: A robot instance.
          gait_generator: Used to query the locomotion phase and leg states.
          state_estimator: Estimate the robot states (e.g. CoM velocity).
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_body_height: The standing height of the robot.
          body_mass: The total mass of the robot.
          body_inertia: The inertia matrix in the body principle frame. We assume
            the body principle coordinate frame has x-forward and z-up.
          num_legs: The number of legs used for force planning.
          friction_coeffs: The friction coeffs on the contact surfaces.
        """
        self._robot_task = robot_task
        self._device = self._robot_task.device
        self._num_envs = self._robot_task.num_envs
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self.desired_speed = torch.as_tensor(
            [desired_speed], device=self._device).repeat(self.num_envs, 1)
        self.desired_twisting_speed = desired_twisting_speed
        self._desired_body_height = desired_body_height
        self._num_legs = 4
        self._friction_coeffs = torch.as_tensor(
            [friction_coeffs], device=self._device).repeat(self.num_envs, 1)

        self._hip_offset = self._hip_offset.unsqueeze(0).repeat(self._num_envs)
        self._default_motor_directions = torch.as_tensor([[-1, -1, -1, -1, 1, 1, 1, 1]], device=self._device).repeat(self._num_envs, 1)

    def reset(self, current_time):
        del current_time

    def update(self, current_time):
        del current_time

    def _estimate_robot_height(self, contacts: Tensor):
        res_desired_body_height = self._desired_body_height

        contacts_indices = torch.nonzero(contacts == 0)

        base_orientation = self._robot_task.root_states[:, 3:7]
        rot_mat = quaternion_to_matrix(base_orientation)

        foot_positions = self._robot_task._footPositionsInBaseFrame()
        foot_positions_world_frame = torch.bmm(
            rot_mat, foot_positions.transpose(-1, -2)).transpose(-1, -2)
        useful_heights = contacts * (-foot_positions_world_frame[..., 2])

        res_desired_body_height[contacts_indices] = useful_heights.sum(
            -1) / contacts.sum(-1)
        return res_desired_body_height

    def mapContactForceToJointTorques(self, contact_force):
        jv = self._robot_task.jacobain_tensor
        all_motor_torques = torch.matmul(contact_force, jv)
        motor_torques = all_motor_torques * self._default_motor_directions
        return motor_torques

    def get_action(self):
        """Computes the torque for stance legs."""
        contacts = self._gait_generator.desired_leg_state == gait_generator_lib.STANCE or self._gait_generator.desired_leg_state.EARLY_CONTACT

        robot_com_position = torch.cat(
            [torch.zeros((self.num_envs, 2), device=self._device), self._estimate_robot_height(contacts)], dim=-1).unsqueeze(0).repeat(self._num_envs, 1)

        robot_com_velocity = self._state_estimator.com_velocity_body_frame
        robot_com_roll_pitch_yaw = self._robot_task._getBaseRollPitchYaw()
        robot_com_roll_pitch_yaw[..., 2] = 0  # To prevent yaw drifting
        robot_com_roll_pitch_yaw_rate = self._robot_task._getBaseRollPitchYawRate()
        robot_q = torch.cat([robot_com_position, robot_com_roll_pitch_yaw], dim=-1)
        robot_dq = torch.cat([robot_com_velocity, robot_com_roll_pitch_yaw_rate], dim=-1)

        # Desired q and dq
        desired_com_position = torch.as_tensor(
            [[0., 0., self._desired_body_height]], device=self._device).repeat(self._num_envs, 1)
        desired_com_velocity = torch.as_tensor(
            [[self.desired_speed[0], self.desired_speed[1], 0.]], device=self._device).repeat(self._num_envs, 1)
        desired_com_roll_pitch_yaw = torch.as_tensor(
            [[0., 0., 0.]], device=self._device).repeat(self._num_envs, 1)
        desired_com_angular_velocity = torch.as_tensor(
            [[0., 0., self.desired_twisting_speed]], device=self._device).repeat(self._num_envs, 1)
        desired_q = torch.cat(
            [desired_com_position, desired_com_roll_pitch_yaw], dim=-1)
        desired_dq = torch.cat(
            [desired_com_velocity, desired_com_angular_velocity], dim=-1)
        # Desired ddq
        desired_ddq = KP * (desired_q - robot_q) + KD * (desired_dq - robot_dq)
        desired_ddq = torch.clamp(desired_ddq, MIN_DDQ, MAX_DDQ)
        contact_forces = qp_torque_optimizer.compute_contact_force(
            self._robot, desired_ddq, contacts=contacts)
        
        # TODO: you are here. You should make the contact forces a Tensor

        motor_torques = self.mapContactForceToJointTorques(contact_forces)
        return motor_torques, contact_forces
