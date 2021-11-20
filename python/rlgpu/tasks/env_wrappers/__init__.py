from rlgpu.tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper
from rlgpu.tasks.env_wrappers.state.plane import PlaneEnvWrapper
from rlgpu.tasks.env_wrappers.state.upstairs import UpstairsEnvWrapper
from rlgpu.tasks.env_wrappers.state.downstairs import DownstairsEnvWrapper
from rlgpu.tasks.env_wrappers.state.mountain_range import MountainRangeEnvWrapper
from rlgpu.tasks.env_wrappers.state.barriers import BarriersEnvWrapper
from rlgpu.tasks.env_wrappers.state.box_ground import BoxGroundEnvWrapper
from rlgpu.tasks.env_wrappers.state.triangle_ground import TriangleGroundEnvWrapper
from rlgpu.tasks.env_wrappers.vision.obstalces import ObstacleEnvWrapper
from rlgpu.tasks.env_wrappers.vision.jumping_stages import JumpingStagesEnvWrapper
from rlgpu.tasks.env_wrappers.vision.jumping_bench import JumpingBenchEnvWrapper
from rlgpu.tasks.env_wrappers.vision.upstairs import UpStairsVisionEnvWrapper
from rlgpu.tasks.env_wrappers.vision.downstairs import DownStairsVisionEnvWrapper
from rlgpu.tasks.env_wrappers.vision.curriculum_way import CurriculumWayEnvWrapper
from rlgpu.tasks.env_wrappers.vision.stepping_stones import SteppingStonesEnvWrapper
from rlgpu.tasks.env_wrappers.vision.stone_ground import StoneGroundEnvWrapper

state_env_wrappers = {"base": BaseEnvWrapper, "plane": PlaneEnvWrapper, "upstairs": UpstairsEnvWrapper,
                      "downstairs": DownstairsEnvWrapper, "mountain_range": MountainRangeEnvWrapper, "barriers": BarriersEnvWrapper, "box_ground": BoxGroundEnvWrapper, "triangle_ground": TriangleGroundEnvWrapper}

vision_env_wrappers = {"obstacles": ObstacleEnvWrapper,
                       "jumping_stages": JumpingStagesEnvWrapper,
                       "jumping_bench": JumpingBenchEnvWrapper,
                       "upstairs_vision": UpStairsVisionEnvWrapper,
                       "downstairs_vision": DownStairsVisionEnvWrapper,
                       "curriculum_way": CurriculumWayEnvWrapper,
                       "stepping_stones": SteppingStonesEnvWrapper,
                       "stone_ground": StoneGroundEnvWrapper}

all_env_wrappers = state_env_wrappers.copy()
all_env_wrappers.update(vision_env_wrappers)


def build_env_wrapper(name, device, cfg):
    return all_env_wrappers[name](device, cfg)
