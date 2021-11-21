from rlgpu.tasks.task_wrappers.base_task_wrapper import BaseTaskWrapper
from rlgpu.tasks.task_wrappers.moving_forward import MovingForwardTask
from rlgpu.tasks.task_wrappers.pace_forward import PaceForwardTask
from rlgpu.tasks.task_wrappers.trot_forward import TrotForwardTask
from rlgpu.tasks.task_wrappers.canter_forward import CanterForwardTask
from rlgpu.tasks.task_wrappers.following_command import FollowingCommandTask

all_task_wrappers = {"base": BaseTaskWrapper,
                     "moving_forward": MovingForwardTask,
                     "pace_forward": PaceForwardTask,
                     "trot_forward": TrotForwardTask,
                     "canter_forward": CanterForwardTask,
                     "following_command": FollowingCommandTask}


def build_task_wrapper(name, device, cfg):
    return all_task_wrappers[name](device, cfg)
