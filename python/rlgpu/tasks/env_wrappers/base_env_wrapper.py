# Environment specific functions that determine:
# - Terrains
# - Barriers
# - Goal positions
# - Randomization of the external environments
# ...

import torch
import trimesh
import numpy as np
from isaacgym import gymapi


class BaseEnvWrapper(object):

    def __init__(self, device, cfg):
        """Initializes the env wrappers."""
        self.cfg = cfg
        self.num_envs = cfg["env"]["numEnvs"]
        # print("num envs", self.num_envs)
        self.device = device
        self.offset = torch.as_tensor(
            cfg["env"]["envOffset"], device=self.device)
        if "envSize" in cfg["env"]:
            self.env_size = cfg["env"]["envSize"]
        else:
            self.env_size = torch.as_tensor(
                (cfg["env"]["envSpacing"], cfg["env"]["envSpacing"]), device=self.device)
        self.ground_type = cfg["env"]["groundType"]["name"]
        self.static_friction = self.cfg["env"]["groundType"]["staticFriction"]
        self.dynamic_friction = self.cfg["env"]["groundType"]["dynamicFriction"]
        self.restitution = self.cfg["env"]["groundType"]["restitution"]
        return

    def check_termination(self, task):
        """Checks if the episode is over."""
        base_pos = task.root_states[task.a1_indices, 0:2]
        flag = torch.all(torch.logical_and(base_pos > -self.offset, base_pos <
                                           self.offset), dim=-1)
        return ~flag

    def create_surroundings(self, task, env_ptr, env_id):
        """Create the surroundings including terrains and obstacles for each environment."""
        handles = []
        i = 0
        if self.ground_type != "plane" or len(self.surrounding_assets) == 0:
            # Currently only support surroundings or trimesh ground
            return handles
        for surrounding_assets, surrounding_cfg in zip(task.surrounding_assets, self.cfg["env"]["surroundings"].values()):
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
            handles.append(handle)
            i += 1
        return handles

    def create_ground(self, task):
        if self.ground_type == "plane":
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = self.static_friction
            plane_params.dynamic_friction = self.dynamic_friction
            task.gym.add_ground(task.sim, plane_params)
            return None
        elif self.ground_type in ["mountain_range"]:
            # Please ensure the loaded 3d model only contains triangle faces
            task.trimesh = trimesh.load(
                "../../assets/terrains/ground/{}.obj".format(self.ground_type))
            vertices = np.asarray(task.trimesh.vertices, dtype=np.float32)
            scale = self.cfg["env"]["groundType"]["scale"]
            offset = self.cfg["env"]["groundType"]["offset"]
            vertices *= np.asarray(scale)
            faces = np.asarray(task.trimesh.faces, dtype=np.uint32)
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = task.trimesh.vertices.shape[0]
            tm_params.nb_triangles = task.trimesh.faces.shape[0]
            tm_params.transform.p.x = offset[0]
            tm_params.transform.p.y = offset[1]
            tm_params.transform.p.z = offset[2]
            task.gym.add_triangle_mesh(task.sim, vertices.flatten(
                order='C'), faces.flatten(order='C'), tm_params)
            np.meshgrid()
            return 
        else:
            raise NotImplementedError
    