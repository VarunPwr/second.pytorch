# Task specific functions that determine:
# - Combination of rewards
# - Terminal conditions
# - Reference data
# ...

class BaseTaskWrapper(object):

    def __init__(self):
        """Initializes the task wrappers."""
        self._draw_ref_model_alpha = 1.
        self._ref_model = -1
        return

    def __call__(self, task):
        return self.reward(task)

    def reset(self, task):
        """Resets the internal state of the task."""
        self._task = task
        return

    def done(self, task):
        """Checks if the episode is over."""
        del task
        return False

    def reward(self, task):
        """Get the reward without side effects."""
        del task
        return 1
