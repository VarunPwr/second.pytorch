# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Sequence, Tuple
from torch import Tensor
from pytorch3d.transforms import quaternion_to_matrix
import torch
from copy import deepcopy

# import time

from mpc_controller import leg_controller
# from mpc_controller import qp_torque_optimizer_cpu as qp_torque_optimizer
from mpc_controller import qp_torque_optimizer


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
            [desired_speed], device=self._device).repeat(self._num_envs, 1)
        self.desired_twisting_speed = torch.as_tensor(
            [desired_twisting_speed], device=self._device).repeat(self._num_envs, 1)
        self._desired_body_height = torch.as_tensor(
            [desired_body_height], device=self._device).repeat(self._num_envs, 1)
        self._num_legs = 4
        self._friction_coeffs = torch.as_tensor(
            [friction_coeffs], device=self._device).repeat(self._num_envs, 1)
        self._default_motor_directions = torch.ones(
            (self._num_envs, 12), device=self._device)
        self.KP = torch.as_tensor((0., 0., 100., 100., 100., 0.), device=self._device).unsqueeze(
            0).repeat(self._num_envs, 1)
        self.KD = torch.as_tensor((40., 30., 10., 10., 10., 30.), device=self._device).unsqueeze(
            0).repeat(self._num_envs, 1)
        self.MAX_DDQ = torch.as_tensor(
            (10., 10., 10., 20., 20., 20.), device=self._device).unsqueeze(0).repeat(self._num_envs, 1)
        self.MIN_DDQ = -self.MAX_DDQ

    def reset(self):
        pass

    def update(self):
        pass

    def _estimate_robot_height(self, contacts: Tensor):
        contacts_indices = torch.stack(
            torch.nonzero(contacts == 1, as_tuple=True))
        if contacts_indices.shape[-1] == 0:
            return self._desired_body_height
        res_desired_body_height = self._desired_body_height
        base_orientation = self._robot_task.root_states[:, 3:7]
        rot_mat = quaternion_to_matrix(base_orientation)

        foot_positions = self._robot_task._footPositionsInBaseFrame()
        foot_positions_world_frame = torch.bmm(
            rot_mat, foot_positions.transpose(-1, -2)).transpose(-1, -2)
        res_desired_body_height = contacts * \
            (-foot_positions_world_frame[..., 2])
        return res_desired_body_height.sum(-1, keepdims=True) / contacts.sum(-1, keepdims=True)

    def mapContactForceToJointTorques(self, contact_force):
        jt = self._robot_task.jacobian_tensor
        jv = jt[:, self._robot_task.feet_indices + 1, :3, :]
        all_motor_torques = torch.bmm(contact_force.view(-1, 1, 3), jv.view(-1, 3, 18)).view(
            self._num_envs, self._num_legs, 18)
        all_motor_torques = all_motor_torques[..., 6:]
        motor_torques = []
        for i in range(self._num_legs):
            motor_torques.append(all_motor_torques[:, i, 3 * i: 3 * (i + 1)])
        motor_torques = torch.cat(motor_torques, dim=-1)
        return motor_torques

    def get_action(self):
        """Computes the torque for stance legs."""
        contacts = torch.logical_or(
            self._gait_generator.desired_leg_state == 1, self._gait_generator.desired_leg_state == 2)

        robot_com_position = torch.cat(
            [torch.zeros((self._num_envs, 2), device=self._device), self._estimate_robot_height(contacts)], dim=-1)

        robot_com_velocity = self._state_estimator.com_velocity_body_frame
        robot_com_roll_pitch_yaw = self._robot_task._getBaseRollPitchYaw()
        robot_com_roll_pitch_yaw[..., 2] = 0
        robot_com_roll_pitch_yaw_rate = self._robot_task._getBaseRollPitchYawRate()
        robot_q = torch.cat(
            [robot_com_position, robot_com_roll_pitch_yaw], dim=-1)
        robot_dq = torch.cat(
            [robot_com_velocity, robot_com_roll_pitch_yaw_rate], dim=-1)
        # Desired q and dq
        zeros_xy = torch.zeros((self._num_envs, 2), device=self._device)

        desired_com_position = torch.cat(
            [zeros_xy, self._desired_body_height], dim=-1)
        desired_com_velocity = deepcopy(self.desired_speed)
        desired_com_velocity[..., -1] = 0
        desired_com_roll_pitch_yaw = torch.zeros(
            (self._num_envs, 3), device=self._device)
        desired_com_angular_velocity = torch.cat(
            [zeros_xy, self.desired_twisting_speed], dim=-1)
        desired_q = torch.cat(
            [desired_com_position, desired_com_roll_pitch_yaw], dim=-1)
        desired_dq = torch.cat(
            [desired_com_velocity, desired_com_angular_velocity], dim=-1)
        # Desired ddq
        desired_ddq = self.KP * (desired_q - robot_q) + \
            self.KD * (desired_dq - robot_dq)
        desired_ddq = torch.minimum(torch.maximum(
            desired_ddq, self.MIN_DDQ), self.MAX_DDQ)
        contact_forces = qp_torque_optimizer.compute_contact_force(
            self._robot_task, desired_ddq, contacts=contacts)
        motor_torques = self.mapContactForceToJointTorques(contact_forces)
        return motor_torques / 100, contact_forces
