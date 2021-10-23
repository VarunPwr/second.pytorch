from rlgpu.tasks.task_wrappers.base_task_wrapper import BaseTaskWrapper

all_task_wrappers = {"base": BaseTaskWrapper, }


def build_task_wrapper(name, device, cfg):
    return all_task_wrappers[name](device, cfg)
