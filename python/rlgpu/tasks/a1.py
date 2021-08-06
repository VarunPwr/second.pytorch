# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
from numpy.core.numeric import indices

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from pytorch3d.transforms import matrix_to_euler_angles, quaternion_to_matrix
import torch
from torch.tensor import Tensor
from typing import Tuple, Dict

terrain_init_pos = {
    "triangle_mesh": [5.2, 0, 0.05],
    "box": [0, 0, 0],
}


class A1(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        # use RL or controller
        self.use_controller = self.cfg["env"]["controller"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["linearVelocityXYRewardScale"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["angularVelocityZRewardScale"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["linearVelocityZRewardScale"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["torqueRewardScale"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["torqueSmoothingRewardScale"] = self.cfg["env"]["learn"]["torqueSmoothingRewardScale"]

        # use diagonal action
        self.diagonal_act = self.cfg["env"]["learn"]["diagonal_act"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.randomize_reward = self.cfg["randomize_reward"]["randomize"]
        self.reward_randomization_params = self.cfg["randomize_reward"]["randomization_params"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # terrain
        self.terrain = self.cfg["env"]["terrain"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        if self.terrain == "triangle_mesh":
            pos[-1] += 0.52

        state = pos + rot + v_lin + v_ang
        self.base_init_state = state

        # sensor settings
        self.historical_step = self.cfg["env"]["sensor"]["historical_step"]
        self.use_sys_information = self.cfg["env"]["sensor"]["sys_id"]

        self.refEnv = self.cfg["env"]["viewer"]["refEnv"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.control_freq_inv = self.cfg["env"]["control"]["controlFrequencyInv"]
        self.dt = sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(
            self.max_episode_length_s / (self.control_freq_inv * self.dt) + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        extra_info_len = 3 if self.use_sys_information else 0
        if self.diagonal_act:
            self.cfg["env"]["numObservations"] = 18 * \
                self.historical_step + 24 + extra_info_len
            self.cfg["env"]["numActions"] = 6
        else:
            self.cfg["env"]["numObservations"] = 24 * \
                (self.historical_step + 1) + extra_info_len
            self.cfg["env"]["numActions"] = 12

        if self.use_controller:
            self.cfg["env"]["numActions"] *= 2

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer is not None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            self.camera_distance = [
                _lookat - _p for _lookat, _p in zip(lookat, p)]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(
            actor_root_state).view(-1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(
            torques).view(self.num_envs, self.num_dof)

        if self.historical_step > 1:
            self.dof_pos_buf = torch.zeros(
                (self.num_envs, self.historical_step, self.num_dof), device=self.device)
            if self.diagonal_act:
                self.actions_buf = torch.zeros(
                    (self.num_envs, self.historical_step, self.num_dof // 2), device=self.device)
            else:
                self.actions_buf = torch.zeros(
                    (self.num_envs, self.historical_step, self.num_dof), device=self.device)
            self.torques_buf = torch.zeros(
                (self.num_envs, self.historical_step, self.num_dof), device=self.device)
        self.commands = torch.zeros(
            self.num_envs, 3 + extra_info_len, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(
            self.num_envs, 3 + extra_info_len)[..., 1]
        self.commands_x = self.commands.view(
            self.num_envs, 3 + extra_info_len)[..., 0]
        self.commands_yaw = self.commands.view(
            self.num_envs, 3 + extra_info_len)[..., 2]

        if self.use_sys_information and self.terrain == "box":
            # the size of box
            self.commands[..., -3] = 0.06
            self.commands[..., -2] = 0.06
            self.commands[..., -1] = 0.04

        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.num_legs = 4
        self._com_offset = - \
            torch.as_tensor([0.012731, 0.002186, 0.000515],
                            device=self.device)
        self._hip_offset = torch.as_tensor([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                                            [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                                            ], device=self.device) + self._com_offset
        self._default_hip_positions = torch.as_tensor([
            [0.17, -0.14, 0],
            [0.17, 0.14, 0],
            [-0.17, -0.14, 0],
            [-0.17, 0.14, 0],
        ]).repeat(self.num_envs, 1)
        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[self.a1_indices] = to_torch(
            self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.time_out_buf = torch.zeros_like(self.reset_buf)

        jt = self.gym.acquire_jacobian_tensor(self.sim, "a1")
        self.jacobian_tensor = gymtorch.wrap_tensor(jt)
        self.feet_dof_pos = self.dof_pos[..., self.feet_indices]
        

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_terrain(self, env_ptr, env_id):

        pose = gymapi.Transform()

        pose.p.x = terrain_init_pos[self.terrain][0]
        pose.p.y = terrain_init_pos[self.terrain][1]
        pose.p.z = terrain_init_pos[self.terrain][2]

        handle = self.gym.create_actor(
            env_ptr, self.terrain_asset, pose, "tm", env_id, 2, 0)
        return handle

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "../../assets"
        asset_file = "urdf/a1/a1.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        a1_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(a1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(a1_asset)
        self.dof_names = self.gym.get_asset_dof_names(a1_asset)
        feet_names = [s for s in body_names if "lower" in s]
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "upper" in s]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(a1_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.a1_indices = []
        self.a1_handles = []
        self.terrain_handles = []
        self.terrain_indices = []
        self.box_handles = []
        self.envs = []

        if self.terrain != "plane":
            terrain_asset_root = "../../assets"
            terrain_asset_file = "terrains/{}/{}.urdf".format(
                self.terrain, self.terrain)
            terrain_asset_path = os.path.join(
                terrain_asset_root, terrain_asset_file)
            terrain_asset_root = os.path.dirname(terrain_asset_path)
            terrain_asset_file = os.path.basename(terrain_asset_path)

            terrain_asset_options = gymapi.AssetOptions()
            terrain_asset_options.fix_base_link = True

            self.terrain_asset = self.gym.load_asset(
                self.sim, terrain_asset_root, terrain_asset_file, terrain_asset_options)

        for i in range(self.num_envs):
            # create env instances
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(
                env_ptr, a1_asset, start_pose, "a1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)
            a1_idx = self.gym.get_actor_index(
                env_ptr, a1_handle, gymapi.DOMAIN_SIM)
            self.a1_indices.append(a1_idx)
            if self.terrain != "plane":
                terrain_handle = self._create_terrain(env_ptr, i)
                terrain_idx = self.gym.get_actor_index(
                    env_ptr, terrain_handle, gymapi.DOMAIN_SIM)
                self.terrain_indices.append(terrain_idx)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], feet_names[i]) - 1
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], knee_names[i]) - 1
        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.a1_handles[0], "trunk")
        self.a1_indices = to_torch(
            self.a1_indices, dtype=torch.long, device=self.device)
        if self.terrain != "plane":
            self.terrain_indices = to_torch(
                self.terrain_indices, dtype=torch.long, device=self.device)


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        if self.historical_step > 1:
            self.actions_buf = torch.cat(
                [self.actions_buf[:, :-1], self.actions.unsqueeze(1)], dim=1)

    def controller_step(self, actions):
        self.actions = actions.clone().to(self.device)
        position_control, torque_control = torch.chunk(self.actions, 2, dim=-1)
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(position_control.contiguous()))
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torque_control.contiguous()))
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
        self.render()
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.post_physics_step()

    def step(self, actions):
        if self.use_controller:
            return self.controller_step(actions)

        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](
                actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        if self.diagonal_act:
            right_action, left_action = torch.chunk(self.actions, 2, dim=-1)
            whole_action = torch.cat(
                [right_action, left_action, left_action, right_action], dim=-1)
            targets_pos = self.action_scale * whole_action + self.default_dof_pos
        else:
            targets_pos = self.action_scale * self.actions + self.default_dof_pos

        # self.gym.set_dof_position_target_tensor(
        #     self.sim, gymtorch.unwrap_tensor(targets_pos))
        self.render()
        # this is the correct way to use action repeat with position control!
        for _ in range(self.control_freq_inv):
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(targets_pos))
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](
                self.obs_buf)

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)
        self._update_viewer()
        if self.historical_step > 1:
            self.dof_pos_buf = torch.cat(
                [self.dof_pos_buf[:, :-1], self.dof_pos.unsqueeze(1)], dim=1)
            self.torques_buf = torch.cat(
                [self.torques_buf[:, :-1], self.torques.unsqueeze(1)], dim=1)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        if self.historical_step > 1:
            self.rew_buf[:], self.reset_buf[:] = compute_a1_reward(
                # tensors
                self.root_states,
                self.commands,
                self.torques,
                self.contact_forces,
                self.knee_indices,
                self.progress_buf,
                # Dict
                self.rew_scales,
                # other
                self.base_index,
                self.max_episode_length,
                last_torques=self.torques_buf[:, -2],
                a1_indices=self.a1_indices
            )
        else:
            self.rew_buf[:], self.reset_buf[:] = compute_a1_reward(
                # tensors
                self.root_states,
                self.commands,
                self.torques,
                self.contact_forces,
                self.knee_indices,
                self.progress_buf,
                # Dict
                self.rew_scales,
                # other
                self.base_index,
                self.max_episode_length,
                last_torques=self.torques,
                a1_indices=self.a1_indices
            )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        if self.historical_step > 1:
            dof_pos = (self.dof_pos_buf -
                       self.default_dof_pos.unsqueeze(1)).view(self.num_envs, -1)
            actions = self.actions_buf.view(self.num_envs, -1)
            self.obs_buf[:] = compute_a1_observations(  # tensors
                self.root_states,
                self.commands,
                dof_pos,
                self.dof_vel,
                self.gravity_vec,
                self.action_scale * actions,
                # scales
                self.lin_vel_scale,
                self.ang_vel_scale,
                self.dof_pos_scale,
                self.dof_vel_scale,
                self.a1_indices
            )
        else:
            self.obs_buf[:] = compute_a1_observations(  # tensors
                self.root_states,
                self.commands,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel,
                self.gravity_vec,
                self.action_scale * self.actions,
                # scales
                self.lin_vel_scale,
                self.ang_vel_scale,
                self.dof_pos_scale,
                self.dof_vel_scale,
                self.a1_indices
            )

    def apply_reward_randomizations(self, rr_params):
        rand_freq = rr_params.get("frequency", 1)

        self.last_step = self.gym.get_frame_count(self.sim)

        do_rew_randomize = (
            self.last_step - self.last_rew_rand_step) >= rand_freq
        if do_rew_randomize:
            self.last_rew_rand_step = self.last_step

        scale_params = rr_params["reward_scale"]
        for k, v in scale_params.items():
            v_range = v["range"]
            self.rew_scales[k] = np.random.uniform(
                low=v_range[0], high=v_range[1]) * self.dt

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if self.randomize_reward:
            self.apply_reward_randomizations(self.reward_randomization_params)

        # positions_offset = torch_rand_float(
        #     0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.05, 0.05,
                                      (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + 0.1 * \
            (torch.rand_like(
                self.default_dof_pos[env_ids], device=self.device) - 0.5)
        self.dof_vel[env_ids] = velocities
        # self.dof_vel[env_ids] = 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        a1_indices = self.a1_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.initial_root_states),
                                                     gymtorch.unwrap_tensor(a1_indices), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(a1_indices), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        if self.historical_step > 1:
            self.dof_pos_buf[env_ids] = 0
            self.actions_buf[env_ids] = 0
            self.torques_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _update_viewer(self):
        if self.viewer is not None:
            lookat = self.root_states[self.a1_indices[self.refEnv], 0:3]
            p = [_lookat - _camera_distance for _camera_distance,
                 _lookat in zip(self.camera_distance, lookat)]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def _foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[...,
                                                 0], angles[..., 1], angles[..., 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(
            l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - \
            torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + \
            torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def _footPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        angles = self.dof_pos
        angles = angles.reshape(self.num_envs, 4, 3)
        foot_positions = torch.zeros_like(angles, device=self.device)
        for i in range(4):
            foot_positions[:, i] = self._foot_position_in_hip_frame(
                angles[:, i], l_hip_sign=(-1)**(i + 1))
        return foot_positions + self._hip_offset

    def _getBaseRollPitchYaw(self):
        """Get minitaur's base orientation in euler angle in the world frame.

        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame.
        """
        base_quat = self.root_states[self.a1_indices, 3:7]
        roll_pitch_yaw = matrix_to_euler_angles(
            quaternion_to_matrix(base_quat))
        return roll_pitch_yaw

    def _getBaseRollPitchYawRate(self):
        quat = self.root_states[self.a1_indices, 3:7]
        ang_vel = quat_rotate_inverse(
            quat, self.root_states[self.a1_indices, 10:13])
        return ang_vel

    def _getHipPositionsInBaseFrame(self):
        return self._default_hip_positions

    def _getContactFootState(self):
        contact_forces = self.contact_forces[:, self.feet_indices].sum(-1)
        contact_states = (contact_forces != 0)
        return contact_states

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_a1_reward(
    # tensors
    root_states: Tensor,
    commands: Tensor,
    torques: Tensor,
    contact_forces: Tensor,
    knee_indices: Tensor,
    episode_lengths: Tensor,
    # Dict
    rew_scales: Dict[str, float],
    # other
    base_index: int,
    max_episode_length: int,
    last_torques: Tensor,
    a1_indices: Tensor
) -> Tuple[Tensor, Tensor]:  # (reward, reset, feet_in air, feet_air_time, episode sums)

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[a1_indices, 3:7]
    base_lin_vel = quat_rotate_inverse(
        base_quat, root_states[a1_indices, 7:10])
    base_ang_vel = quat_rotate_inverse(
        base_quat, root_states[a1_indices, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(
        commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * \
        rew_scales["linearVelocityXYRewardScale"]
    rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * \
        rew_scales["angularVelocityZRewardScale"]
    # z velocity penalty
    rew_z_vel = torch.square(
        base_lin_vel[:, 2]) * rew_scales["linearVelocityZRewardScale"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * \
        rew_scales["torqueRewardScale"]
    if last_torques is not None:
        rew_torque += torch.sum(torch.square(torques - last_torques),
                                dim=1) * rew_scales["torqueSmoothingRewardScale"]
    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_z_vel
    total_reward = torch.clip(total_reward, 0., None)
    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(
        contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)

    # reset due to fall
    reset = reset | (base_quat[:, -1] < 0.6)

    # no terminal reward for time-outs
    time_out = episode_lengths > max_episode_length
    reset = reset | time_out

    return total_reward.detach(), reset


@torch.jit.script
def compute_a1_observations(root_states: Tensor,
                            commands: Tensor,
                            dof_pos: Tensor,
                            dof_vel: Tensor,
                            gravity_vec: Tensor,
                            actions: Tensor,
                            lin_vel_scale: float,
                            ang_vel_scale: float,
                            dof_pos_scale: float,
                            dof_vel_scale: float,
                            a1_indices: Tensor
                            ) -> Tensor:

    # base_position = root_states[:, 0:3]
    base_quat = root_states[a1_indices, 3:7]
    base_lin_vel = quat_rotate_inverse(
        base_quat, root_states[a1_indices, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(
        base_quat, root_states[a1_indices, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)

    dof_pos_scaled = dof_pos * dof_pos_scale

    commands_scaled = commands[..., :3] * \
        torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale],
                     requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     commands[..., 3:],
                     dof_pos_scaled,
                     dof_vel * dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs
