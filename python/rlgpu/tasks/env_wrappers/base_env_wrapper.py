# Environment specific functions that determine:
# - Terrains
# - Barriers
# - Goal positions
# - Randomization of the external environments
# ...

import torch
from isaacgym import gymapi


class BaseEnvWrapper(object):

    def __init__(self, device, cfg):
        """Initializes the env wrappers."""
        self.cfg = cfg
        self.num_envs = cfg["env"]["numEnvs"]
        self.device = device
        self.offset = torch.as_tensor(
            cfg["env"]["envOffset"], device=self.device)
        if "envSize" in cfg["env"]:
            self.env_size = cfg["env"]["envSize"]
        else:
            self.env_size = torch.as_tensor(
                (cfg["env"]["envSpacing"], cfg["env"]["envSpacing"]), device=self.device)

        return

    def check_termination(self, task):
        """Checks if the episode is over."""
        base_pos = task.root_states[task.a1_indices, 0:2]
        flag = torch.all((base_pos > self.offset and base_pos <
                          self.offset + self.env_size), dim=-1)
        return ~flag

    def create_surroundings(self, task, env_ptr, env_id):
        """Create the surroundings including terrains and obstacles for each environment."""
        handles = []
        i = 0
        for surrounding_assets, surrounding_cfg in zip(task.surrounding_assets, self.cfg["env"]["surroundings"]):
            pose = gymapi.Transform()
            pose.p.x = surrounding_cfg["surrounding_origin"][0]
            pose.p.y = surrounding_cfg["surrounding_origin"][1]
            pose.p.z = surrounding_cfg["surrounding_origin"][2]

            handle = task.gym.create_actor(
                env_ptr, surrounding_assets, pose, "sr_{}".format(i), env_id, 2 + i, 0)
            if surrounding_cfg["texture"] != "none":
                th = task.gym.create_texture_from_file(
                    task.sim, "../../../assets/textures/{}".format(surrounding_cfg["texture"]))
                task.gym.set_rigid_body_texture(
                    env_ptr, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, th)

            i += 1

        return handles
