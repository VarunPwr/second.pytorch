# Task specific functions that determine:
# - Combination of rewards
# - Terminal conditions
# - Reference data
# ...
import os
import yaml


class BaseTaskWrapper(object):

    def __init__(self, device, cfg):
        """Initializes the task wrappers."""
        self.device = device
        self.cfg = cfg
        self.task_name = cfg["task"]["name"]
        self.file_name = self.task_name + '.yaml'
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', self.file_name), 'r') as f:
            self.task_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return

    def check_termination(self, task):
        """Checks if the episode is over."""
        del task
        return False
