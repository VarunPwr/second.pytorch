"""Gait pattern planning module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum


class LegState(enum.Enum):
    """The state of a leg during locomotion."""
    SWING = 0
    STANCE = 1
    EARLY_CONTACT = 2
    LOSE_CONTACT = 3


class GaitGenerator(object):
    """Generates the leg swing/stance pattern for the robot."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, current_time, contact_state):
        pass
