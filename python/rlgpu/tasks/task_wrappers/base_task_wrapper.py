# Task specific functions that determine:
# - Combination of rewards
# - Terminal conditions
# - Reference data
# ...

class BaseTaskWrapper(object):

    def __init__(self, device, cfg):
        """Initializes the task wrappers."""
        self.device = device
        self.cfg = cfg
        return

    def __call__(self, task):
        return self.reward(task)

    def check_termination(self, task):
        """Checks if the episode is over."""
        del task
        return False

    def reward(self, task):
        """Get the reward without side effects."""
        del task
        return 1
