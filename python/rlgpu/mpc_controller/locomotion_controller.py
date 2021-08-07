"""A model based controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Callable
import numpy as np
import time
import os
import inspect
import torch
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class LocomotionController(object):
    """Generates the quadruped locomotion.

    The actual effect of this controller depends on the composition of each
    individual subcomponent.

    """

    def __init__(
        self,
        robot_task,
        gait_generator,
        state_estimator,
        swing_leg_controller,
        stance_leg_controller,
    ):
        """Initializes the class.

        Args:
          robot: A robot instance.
          gait_generator: Generates the leg swing/stance pattern.
          state_estimator: Estimates the state of the robot (e.g. center of mass
            position or velocity that may not be observable from sensors).
          swing_leg_controller: Generates motor actions for swing legs.
          stance_leg_controller: Generates motor actions for stance legs.
          clock: A real or fake clock source.
        """
        self._robot_task = robot_task
        self._device = self._robot_task.device
        self._num_envs = self._robot_task.num_envs
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self._swing_leg_controller = swing_leg_controller
        self._stance_leg_controller = stance_leg_controller

    @property
    def swing_leg_controller(self):
        return self._swing_leg_controller

    @property
    def stance_leg_controller(self):
        return self._stance_leg_controller

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def state_estimator(self):
        return self._state_estimator

    def reset(self):
        self._gait_generator.reset()
        self._state_estimator.reset()
        self._swing_leg_controller.reset()
        self._stance_leg_controller.reset()

    def update(self):
        current_time = self._robot_task.progress_buf
        self._gait_generator.update(current_time)
        self._state_estimator.update()
        self._swing_leg_controller.update()
        self._stance_leg_controller.update()

    def get_action(self):
        """Returns the control ouputs (e.g. positions/torques) for all motors."""
        swing_foot_position, swing_foot_indices = self._swing_leg_controller.get_action()
        motor_torque, qp_sol = self._stance_leg_controller.get_action()
        position_control = torch.zeros_like(motor_torque)
        position_control[:, swing_foot_indices] = swing_foot_position.flatten(1)[:, swing_foot_indices]
        hybrid_control = torch.cat([position_control, motor_torque], dim=-1)
        return hybrid_control, dict(qp_sol=qp_sol)
