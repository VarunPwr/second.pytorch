from rlgpu.tasks.utilizers.state_randomization import StateRandomizer
from rlgpu.tasks.utilizers.reward_randomization import RewardRandomizer
from rlgpu.tasks.utilizers.env_curriculum import EnvCurriculumScheduler

all_utilizers = {"randomize_reward": RewardRandomizer,
                 "randomize_state": StateRandomizer, "env_curriculum": EnvCurriculumScheduler}


def build_utilizer(name, cfg):
    return all_utilizers[name](cfg)
