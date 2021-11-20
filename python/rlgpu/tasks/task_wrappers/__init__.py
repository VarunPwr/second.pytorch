from rlgpu.tasks.task_wrappers.base_task_wrapper import BaseTaskWrapper
from rlgpu.tasks.task_wrappers.moving_forward import MovingForwardTask
from rlgpu.tasks.task_wrappers.following_command import FollowingCommandTask

all_task_wrappers = {"base": BaseTaskWrapper,
                     "moving_forward": MovingForwardTask,
                     "following_command": FollowingCommandTask}


def build_task_wrapper(name, device, cfg):
    return all_task_wrappers[name](device, cfg)
