"""Gait pattern planning module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Sequence
from torch import Tensor
from copy import deepcopy
import torch

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


from mpc_controller import gait_generator

A1_TROTTING = [
    0,
    1,
    1,
    0,
]

_NOMINAL_STANCE_DURATION = 0.032
_NOMINAL_DUTY_FACTOR = 0.5
_NOMINAL_CONTACT_DETECTION_PHASE = 0.1


class OpenloopGaitGenerator(gait_generator.GaitGenerator):
    """Generates openloop gaits for quadruped robots.

    A flexible open-loop gait generator. Each leg has its own cycle and duty
    factor. And the state of each leg alternates between stance and swing. One can
    easily formuate a set of common quadruped gaits like trotting, pacing,
    pronking, bounding, etc by tweaking the input parameters.
    """

    def __init__(
        self,
        robot_task: Any,
        stance_duration: float = _NOMINAL_STANCE_DURATION,
        duty_factor: float = _NOMINAL_DUTY_FACTOR,
        initial_leg_state: Sequence[int] = A1_TROTTING,
        initial_leg_phase: Sequence[float] = [0.9, 0, 0, 0.9],
        contact_detection_phase_threshold:
        float = _NOMINAL_CONTACT_DETECTION_PHASE,
    ):
        """Initializes the class.

        Args:
          robot: A quadruped robot that at least implements the GetFootContacts API
            and num_legs property.
          stance_duration: The desired stance duration.
          duty_factor: The ratio  stance_duration / total_gait_cycle.
          initial_leg_state: The desired initial swing/stance state of legs indexed
            by their id.
          initial_leg_phase: The desired initial phase [0, 1] of the legs within the
            full swing + stance cycle.
          contact_detection_phase_threshold: Updates the state of each leg based on
            contact info, when the current normalized phase is greater than this
            threshold. This is essential to remove false positives in contact
            detection when phase switches. For example, a swing foot at at the
            beginning of the gait cycle might be still on the ground.
        """
        self._robot_task = robot_task
        self._num_legs = self._robot_task.num_legs
        self._num_envs = self._robot_task.num_envs
        self._device = self._robot_task.device

        # make them into tensor, repeat for each env
        self._stance_duration = torch.as_tensor(
            [stance_duration], device=self._device).repeat(self._num_envs, 4)
        self._duty_factor = torch.as_tensor(
            [duty_factor], device=self._device).repeat(self._num_envs, 4)
        self._swing_duration = self._stance_duration / \
            self._duty_factor - self._stance_duration
        if len(initial_leg_phase) != self._num_legs:
            raise ValueError(
                "The number of leg phases should be the same as number of legs.")
        self._initial_leg_phase = torch.as_tensor(
            [initial_leg_phase], device=self._device).repeat(self._num_envs, 1)
        if len(initial_leg_state) != self._num_legs:
            raise ValueError(
                "The number of leg states should be the same of number of legs.")
        self._initial_leg_state = torch.as_tensor(
            [initial_leg_state], device=self._device).repeat(self._num_envs, 1)
        self._next_leg_state = torch.zeros_like(
            self._initial_leg_state, device=self._device)
        # The ratio in cycle is duty factor if initial state of the leg is STANCE,
        # and 1 - duty_factory if the initial state of the leg is SWING.
        self._initial_state_ratio_in_cycle = torch.zeros_like(
            self._initial_leg_state, device=self._device)

        self._initial_state_ratio_in_cycle = ((1 - duty_factor) * (self._initial_leg_state == 0) + duty_factor * (self._initial_leg_state != 0)).float()
        self._next_leg_state = (self._initial_leg_state == 0).long()

        self._contact_detection_phase_threshold = contact_detection_phase_threshold

        # The normalized phase within swing or stance duration.
        self._normalized_phase = None
        self._leg_state = None
        self._desired_leg_state = None

        self.reset()

    def reset(self):
        # The normalized phase within swing or stance duration.
        self._normalized_phase = torch.zeros(
            (self._num_envs, self._num_legs), device=self._device)
        self._leg_state = deepcopy(self._initial_leg_state)
        self._desired_leg_state = deepcopy(self._initial_leg_state)

    @property
    def desired_leg_state(self) -> Tensor:
        """The desired leg SWING/STANCE states.

        Returns:
          The SWING/STANCE states for all legs.

        """
        return self._desired_leg_state

    @property
    def leg_state(self) -> Tensor:
        """The leg state after considering contact with ground.

        Returns:
          The actual state of each leg after accounting for contacts.
        """
        return self._leg_state

    @property
    def swing_duration(self) -> Tensor:
        return self._swing_duration

    @property
    def stance_duration(self) -> Tensor:
        return self._stance_duration

    @property
    def normalized_phase(self) -> Tensor:
        """The phase within the current swing or stance cycle.

        Reflects the leg's phase within the current swing or stance stage. For
        example, at the end of the current swing duration, the phase will
        be set to 1 for all swing legs. Same for stance legs.

        Returns:
          Normalized leg phase for all legs.

        """
        return self._normalized_phase

    def update(self, current_time: Tensor):
        contact_state = self._robot_task._getContactFootState()
        full_cycle_period = self._stance_duration / self._duty_factor
        augmented_time = current_time.unsqueeze(-1) + self._initial_leg_phase * full_cycle_period
        phase_in_full_cycle = torch.fmod(augmented_time,
                                         full_cycle_period) / full_cycle_period
        ratio = self._initial_state_ratio_in_cycle
        indices = torch.nonzero(phase_in_full_cycle < ratio, as_tuple=True)
        non_indices = torch.nonzero(phase_in_full_cycle >= ratio, as_tuple=True)
        self._desired_leg_state[indices] = self._initial_leg_state[indices]
        self._normalized_phase[indices] = (phase_in_full_cycle / ratio)[indices]
        self._desired_leg_state[non_indices] = self._next_leg_state[non_indices]
        self._normalized_phase[non_indices] = ((
            phase_in_full_cycle - ratio) / (1 - ratio))[non_indices]
        self._leg_state = self._desired_leg_state
        early_contact = torch.logical_and(torch.logical_and(self._normalized_phase > self._contact_detection_phase_threshold, torch.eq(self._leg_state, 0)), contact_state)
        early_contact_indices = torch.stack(torch.nonzero(early_contact, as_tuple=True))
        if early_contact_indices.shape[-1] > 0:
            self._leg_state[torch.nonzero(early_contact, as_tuple=True)] = 2
        lost_contact = torch.logical_and(torch.logical_and(self._normalized_phase > self._contact_detection_phase_threshold, torch.eq(self._leg_state, 1)), (0 == contact_state))
        lost_contact_indices = torch.stack(torch.nonzero(lost_contact, as_tuple=True))
        if lost_contact_indices.shape[-1] > 0:
            self._leg_state[torch.nonzero(lost_contact, as_tuple=True)] = 3