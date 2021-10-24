from rlgpu.tasks.utilizers.state_randomization import StateRandomizer
from rlgpu.tasks.utilizers.reward_randomization import RewardRandomizer

all_utilizers = {"randomize_reward": RewardRandomizer, "randomize_state": StateRandomizer}


def build_utilizer(name, cfg):
    return all_utilizers[name](cfg)
