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
        device: str,
        gait_generator: Any,
        state_estimator: Any,
        num_envs: int,
        num_legs: int = 4,
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
        self._device = device
        self._num_envs = num_envs
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self.desired_speed = torch.as_tensor(
            [desired_speed], device=self._device).repeat(self.num_envs, 1)
        self.desired_twisting_speed = desired_twisting_speed
        self._desired_body_height = desired_body_height
        self._num_legs = num_legs
        self._friction_coeffs = torch.as_tensor(
            [friction_coeffs], device=self._device).repeat(self.num_envs, 1)

    def reset(self, current_time):
        del current_time

    def update(self, current_time):
        del current_time

    def _estimate_robot_height(self, robot_states: Tensor, contacts: Tensor, foot_positions: Tensor):
        res_desired_body_height = deepcopy(self._desired_body_height)

        contacts_indices = torch.nonzero(contacts == 0)

        base_orientation = robot_states[:, 3:7]
        rot_mat = quaternion_to_matrix(base_orientation)

        foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T

        foot_positions_world_frame = torch.bmm(
            rot_mat, foot_positions.transpose(-1, -2)).transpose(-1, -2)
        # pylint: disable=unsubscriptable-object
        useful_heights = contacts * (-foot_positions_world_frame[..., 2])

        res_desired_body_height[contacts_indices] = useful_heights.sum(
            -1) / contacts.sum(-1)
        return res_desired_body_height

    def compute_jacobian(self, robot, link_id):
        """Computes the Jacobian matrix for the given link.

        Args:
          robot: A robot instance.
          link_id: The link id as returned from loadURDF.

        Returns:
          The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
          robot. For a quadruped, the first 6 columns of the matrix corresponds to
          the CoM translation and rotation. The columns corresponds to a leg can be
          extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
        """
        all_joint_angles = [state[0] for state in robot._joint_states]
        zero_vec = [0] * len(all_joint_angles)
        jv, _ = self.pybullet_client.calculateJacobian(robot.quadruped, link_id,
                                                       (0, 0, 0), all_joint_angles,
                                                       zero_vec, zero_vec)
        jacobian = np.array(jv)
        assert jacobian.shape[0] == 3
        return jacobian

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        assert len(self._foot_link_ids) == self.num_legs
        return self.compute_jacobian(
            robot=self,
            link_id=self._foot_link_ids[leg_id],
        )

    def MapContactForceToJointTorques(self, leg_id, contact_force):
        jv = self.ComputeJacobian(leg_id)
        all_motor_torques = np.matmul(contact_force, jv)
        motor_torques = {}
        motors_per_leg = self.num_motors // self.num_legs
        com_dof = 6
        for joint_id in range(leg_id * motors_per_leg,
                              (leg_id + 1) * motors_per_leg):
            motor_torques[joint_id] = all_motor_torques[
                com_dof + joint_id] * self._motor_direction[joint_id]

        return motor_torques

    def get_action(self):
        """Computes the torque for stance legs."""
        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_generator.desired_leg_state],
            dtype=np.int32)

        robot_com_position = np.array(
            (0., 0., self._estimate_robot_height(contacts)))
        robot_com_velocity = self._state_estimator.com_velocity_body_frame
        robot_com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw())
        robot_com_roll_pitch_yaw[2] = 0  # To prevent yaw drifting
        robot_com_roll_pitch_yaw_rate = self._robot.GetBaseRollPitchYawRate()
        robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
        robot_dq = np.hstack(
            (robot_com_velocity, robot_com_roll_pitch_yaw_rate))

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
        # TODO: you are here
        action = {}
        for leg_id, force in enumerate(contact_forces):
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            # if self._gait_generator.leg_state[
            #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
            #   force = (0, 0, 0)
            motor_torques = self.MapContactForceToJointTorques(
                leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = (0, 0, 0, 0, torque)
        return action, contact_forces
