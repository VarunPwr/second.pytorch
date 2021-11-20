from rlgpu.tasks.learnable.gail import GAIL

all_learners = {"gail": GAIL}


def build_learner(name, cfg):
    return all_learners[name](cfg)
