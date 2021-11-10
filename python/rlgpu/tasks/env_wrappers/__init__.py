from rlgpu.tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper
from rlgpu.tasks.env_wrappers.state.plane import PlaneEnvWrapper
from rlgpu.tasks.env_wrappers.state.upstairs import UpstairsEnvWrapper
from rlgpu.tasks.env_wrappers.state.downstairs import DownstairsEnvWrapper
from rlgpu.tasks.env_wrappers.state.mountain_range import MountainRangeEnvWrapper
from rlgpu.tasks.env_wrappers.state.barriers import BarriersEnvWrapper

state_env_wrappers = {"base": BaseEnvWrapper, "plane": PlaneEnvWrapper, "upstairs": UpstairsEnvWrapper,
                      "downstairs": DownstairsEnvWrapper, "mountain_range": MountainRangeEnvWrapper, "barriers": BarriersEnvWrapper}

vision_env_wrappers = {}

all_env_wrappers = state_env_wrappers.copy()
all_env_wrappers.update(vision_env_wrappers)


def build_env_wrapper(name, device, cfg):
    return all_env_wrappers[name](device, cfg)
