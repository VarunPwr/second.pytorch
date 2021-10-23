from rlgpu.tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper

all_env_wrappers = {"base": BaseEnvWrapper, }


def build_env_wrapper(name, device, cfg):
    return all_env_wrappers[name](device, cfg)
