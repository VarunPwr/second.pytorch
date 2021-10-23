import torch

class ForwardTask(object):
    """Default empy task."""

    def __init__(self):
        """Initializes the task."""
        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)

    def __call__(self, task):
        return self.reward(task)

    def reset(self, task):
        """Resets the internal state of the task."""
        self._task = task
        self.last_base_pos = task.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos

    def update(self, task):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = task.robot.GetBasePosition()

    def done(self, task):
        """Checks if the episode is over.

           If the robot base becomes unstable (based on orientation), the episode
           terminates early.
        """
        del task
        return

    def reward(self, task):
        """Get the reward without side effects."""
        del task
        return self.current_base_pos[0] - self.last_base_pos[0]
