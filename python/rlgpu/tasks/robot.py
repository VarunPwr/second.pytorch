# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import sys
from rlgpu.utils.torch_jit_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from torch import Tensor
from rlgpu.tasks.task_wrappers import build_task_wrapper
from rlgpu.tasks.env_wrappers import build_env_wrapper
from rlgpu.tasks.utilizers import build_utilizer
from rlgpu.tasks.learner import build_learner


class Robot:

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.gym = gymapi.acquire_gym()

        self.cfg = cfg
        self.task_name = cfg["task"]["name"]
        self.env_name = cfg["env"]["name"]
        self.num_envs = cfg["env"]["numEnvs"]
        self.device_type = device_type
        self.device_id = device_id

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self._prepare_wrappers()
        self.get_image = self.env_wrapper.env_cfg["vision"]["get_image"]
        # double check!
        self.graphics_device_id = self.device_id
        self.headless = headless
        if self.headless and not self.get_image:
            self.graphics_device_id = -1

        self.headless = cfg["headless"]

        # self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self._register()
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self._build_viewer()
        self._build_buf()
        self._build_utilizers()
        self._prepare_reward_function()
        self._build_learners()
        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):

        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        # self._create_ground_plane()
        self.env_wrapper.load_surrounding_assets(self)

        self._sample_init_state()
        self._create_ground()
        self._create_envs(self.cfg["env"]['envSpacing'],
                          int(np.sqrt(self.num_envs)))

    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def _create_ground(self):
        self.robot_origin = self.env_wrapper.create_ground(
            self) + torch.as_tensor(self.base_init_state[:3], device=self.device)
        # self.robot_origin = torch.zeros((self.num_envs, 3), device=self.device)
        # self.robot_origin = self.env_wrapper.create_ground(self)

    def _register(self):

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.hip_action_scale = self.cfg["env"]["control"]["hipActionScale"]
        if self.get_image:
            # vision-guided mode
            self.frame_stack = self.env_wrapper.env_cfg["vision"]["frame_stack"]
            self.vision_update_freq = self.env_wrapper.env_cfg["vision"]["update_freq"]
            self.image_type = self.env_wrapper.env_cfg["vision"]["image_type"]
            self.width = self.env_wrapper.env_cfg["vision"]["width"]
            self.height = self.env_wrapper.env_cfg["vision"]["height"]
            self.camera_angle = self.env_wrapper.env_cfg["vision"]["camera_angle"]
        self.frame_count = 0

        # use diagonal action
        self.diagonal_act = self.cfg["env"]["learn"]["diagonal_act"]

        # commands
        self.command_type = self.cfg["env"]["command"]
        self.command_change_step = self.cfg["env"]["commandChangeStep"]

        # sensor settings
        self.historical_step = self.cfg["env"]["sensor"]["historical_step"]
        self.use_sys_information = self.cfg["env"]["sensor"]["sys_id"]

        self.risk_reward = self.cfg["env"]["risk_reward"]
        self.rush_reward = self.cfg["env"]["rush_reward"]
        self.vel_reward_exp_coeff = self.cfg["env"]["vel_reward_exp_coeff"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.control_freq_inv = self.cfg["env"]["control"]["controlFrequencyInv"]
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(
            self.max_episode_length_s / (self.control_freq_inv * self.dt) + 0.5)

        extra_info_len = 3 if self.use_sys_information else 0
        if self.diagonal_act:
            self.cfg["env"]["numObservations"] = 18 * \
                self.historical_step + 24 + extra_info_len + 12
            self.cfg["env"]["numActions"] = 6
        else:
            self.cfg["env"]["numObservations"] = 24 * \
                (self.historical_step + 1) + extra_info_len + 12
            self.cfg["env"]["numActions"] = 12
        self.state_obs_size = self.cfg["env"]["numObservations"]
        if self.get_image:
            if self.image_type == "depth":
                image_obs_size = self.width * \
                    self.height * self.frame_stack
            elif self.image_type == "rgb":
                image_obs_size = self.width * \
                    self.height * self.frame_stack * 3
            elif self.image_type == "rgbd":
                image_obs_size = self.width * \
                    self.height * self.frame_stack * 4
            else:
                raise NotImplementedError

            self.cfg["env"]["numObservations"] += image_obs_size
            self.image_obs_size = image_obs_size

        self.num_obs = self.cfg["env"]["numObservations"]
        self.num_states = self.cfg["env"].get("numStates", 0)
        self.num_actions = self.cfg["env"]["numActions"]

    def _build_utilizers(self):
        self.randomizer = {}
        self.curriculum_scheduler = {}
        if self.cfg["randomize_state"]["randomize"]:
            self.randomizer["randomize_state"] = build_utilizer(
                "randomize_state", self.cfg)
            self.randomize_input = True
        if self.cfg["randomize_reward"]["randomize"]:
            self.randomizer["randomize_reward"] = build_utilizer(
                "randomize_reward", self.cfg)

    def _build_learners(self):
        self.learners = {}
        if "learners" not in self.cfg:
            return
        for key, _ in self.cfg["learners"].items():
            self.learners[key] = build_learner(
                key, self.num_envs, self.device,  self.cfg)

    def _build_buf(self):
        # get gym state tensors
        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        self.dr_randomizations = {}

        self.last_step = -1
        self.last_rand_step = -1
        self.last_rew_rand_step = -1

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
        self.last_root_states = torch.zeros_like(self.root_states)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.last_dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.last_dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]

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
                self.last_actions = torch.zeros(
                    self.num_envs, self.num_dof // 2, dtype=torch.float, device=self.device, requires_grad=False)
            else:
                self.actions_buf = torch.zeros(
                    (self.num_envs, self.historical_step, self.num_dof), device=self.device)
                self.last_actions = torch.zeros(

                    self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            self.torques_buf = torch.zeros(
                (self.num_envs, self.historical_step, self.num_dof), device=self.device)

        if self.get_image:
            if self.frame_stack > 1:
                if self.image_type == "depth":
                    self.image_buf = torch.zeros(
                        (self.num_envs, self.frame_stack, self.width * self.height), device=self.device
                    )
                elif self.image_type == "rgb":
                    self.image_buf = torch.zeros(
                        (self.num_envs, self.frame_stack, self.width * self.height * 3), device=self.device
                    )
                elif self.image_type == "rgbd":
                    self.image_buf = torch.zeros(
                        (self.num_envs, self.frame_stack, self.width * self.height * 4), device=self.device
                    )
                else:
                    raise NotImplementedError

        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(
            self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(
            self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(
            self.num_envs, 3)[..., 2]

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
        ], device=self.device)
        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[self.a1_indices] = to_torch(
            self.init_states_for_each_env, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.time_out_buf = torch.zeros_like(self.reset_buf)

        self.feet_dof_pos = self.dof_pos[..., self.feet_indices]
        if self.diagonal_act:
            self.action_scale = torch.as_tensor(
                [self.hip_action_scale, self.action_scale, self.action_scale] * 2, device=self.device)
        else:
            self.action_scale = torch.as_tensor(
                [self.hip_action_scale, self.action_scale, self.action_scale] * 4, device=self.device)

    def _sample_init_state(self):
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        pos[0] += self.cfg["env"]["robot_origin"][0]
        pos[1] += self.cfg["env"]["robot_origin"][1]
        pos[2] += self.cfg["env"]["robot_origin"][2]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state
        self.init_states_for_each_env = torch.as_tensor(
            [self.base_init_state for _ in range(self.num_envs)], device=self.device)

    def _prepare_motor_params(self):

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

    def _prepare_wrappers(self):

        self.task_wrapper = build_task_wrapper(
            self.task_name, self.device, self.cfg)
        self.env_wrapper = build_env_wrapper(
            self.env_name, self.device, self.cfg)

    def _create_envs(self, spacing, num_per_row):
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
        dof_props_asset = self.gym.get_asset_dof_properties(a1_asset)
        self.num_dof = self.gym.get_asset_dof_count(a1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(a1_asset)
        self.dof_names = self.gym.get_asset_dof_names(a1_asset)
        feet_names = [(i, s) for i, s in enumerate(body_names) if "lower" in s]
        self.feet_indices_in_bodies = torch.as_tensor(
            [fn[0] for fn in feet_names], device=self.device)
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "upper" in s]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.a1_indices = []
        self.a1_handles = []
        self.surrounding_indices = []
        self.envs = []
        if self.get_image:
            self.camera_handles = []
            self.depth_image = []

        self._prepare_motor_params()
        self.init_states_for_each_env[..., :3] = self.robot_origin
        for i in range(self.num_envs):
            # create env instances
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row)
            if self.robot_origin is not None:
                start_pose.p = gymapi.Vec3(*self.robot_origin[i])
            a1_handle = self.gym.create_actor(
                env_ptr, a1_asset, start_pose, "a1", i, 1, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)

            self.gym.set_actor_dof_properties(
                env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)
            a1_idx = self.gym.get_actor_index(
                env_ptr, a1_handle, gymapi.DOMAIN_SIM)
            self.a1_indices.append(a1_idx)
            if self.env_wrapper.surroundings is not None:
                surrounding_handles = self.env_wrapper.create_surroundings(
                    self, env_ptr, i)
                surrounding_indices = [self.gym.get_actor_index(
                    env_ptr, sh, gymapi.DOMAIN_SIM) for sh in surrounding_handles]
                self.surrounding_indices.append(surrounding_indices)
            if self.get_image:
                camera_properties = gymapi.CameraProperties()
                camera_properties.width = self.width
                camera_properties.height = self.height
                camera_properties.enable_tensors = True
                head_camera = self.gym.create_camera_sensor(
                    env_ptr, camera_properties)
                camera_offset = gymapi.Vec3(0.3, 0, 0)
                camera_rotation = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 1, 0), np.deg2rad(self.camera_angle))
                body_handle = self.gym.get_actor_rigid_body_handle(
                    env_ptr, a1_handle, 0)
                self.gym.attach_camera_to_body(head_camera, env_ptr, body_handle, gymapi.Transform(
                    camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(head_camera)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], feet_names[i][1]) - 1
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], knee_names[i]) - 1
        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.a1_handles[0], "trunk")
        self.a1_indices = to_torch(
            self.a1_indices, dtype=torch.long, device=self.device)
        penalized_contact_names = []
        for name in self.cfg["env"]["asset"]["penalize_contacts_on"]:
            penalized_contact_names.extend(
                [s for s in body_names if name in s])
        self.penalised_contact_indices = torch.zeros(len(
            penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], penalized_contact_names[i])
        termination_contact_names = []
        for name in self.cfg["env"]["asset"]["terminate_after_contacts_on"]:
            termination_contact_names.extend(
                [s for s in body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(
            termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], termination_contact_names[i])
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.surrounding_indices = to_torch(
            self.surrounding_indices, dtype=torch.long, device=self.device)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.last_root_states = self.root_states.clone()
        self.last_dof_pos = self.dof_pos.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_actions[:] = self.actions[:]
        if self.historical_step > 1:
            self.actions_buf = torch.cat(
                [self.actions_buf[:, 1:], self.actions.unsqueeze(1)], dim=1)

    def controller_step(self, actions):
        self.actions = actions.clone().to(self.device)
        position_control, torque_control = torch.chunk(self.actions, 2, dim=-1)
        position_control += self.default_dof_pos
        torque_control *= 100
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(position_control.contiguous()))
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torque_control.contiguous()))
        # self.gym.set_dof_position_target_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(position_control.contiguous()))
        # self.gym.set_dof_actuation_force_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(torque_control.contiguous()))

        # self.gym.set_dof_position_target_tensor(
        #     self.sim, gymtorch.unwrap_tensor(torch.zeros_like(position_control)))
        # self.gym.set_dof_actuation_force_tensor(
        #     self.sim, gymtorch.unwrap_tensor(torch.zeros_like(torque_control)))
        # for _ in range(self.control_freq_inv):
        #     self.gym.simulate(self.sim)
        self.gym.simulate(self.sim)

        self.render()
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.post_physics_step()

    def get_states(self):
        return self.states_buf

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def step(self, actions):
        if self.randomize_input:
            actions = self.randomizer["randomize_state"].state_randomizations['actions']['noise_lambda'](
                actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        if self.diagonal_act:
            right_action, left_action = torch.chunk(
                self.action_scale * self.actions, 2, dim=-1)
            whole_action = torch.cat(
                [right_action, left_action, left_action, right_action], dim=-1)
            targets_pos = whole_action + self.default_dof_pos
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

        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.randomize_input:
            self.obs_buf = self.randomizer["randomize_state"].state_randomizations['observations']['noise_lambda'](
                self.obs_buf)

    def post_physics_step(self):
        self.frame_count += 1
        self.progress_buf += 1
        self.base_quat = self.root_states[self.a1_indices, 3:7]
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[self.a1_indices, 7:10]) * self.lin_vel_scale
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[self.a1_indices, 10:13]) * self.ang_vel_scale
        change_commmand_env_ids = (torch.fmod(
            self.progress_buf, self.command_change_step) == 0).float().nonzero(as_tuple=False).squeeze(-1)
        if self.task_wrapper.task_name == "following_command" and len(change_commmand_env_ids) > 0:
            self.reset_command(change_commmand_env_ids)
        self.task_wrapper.check_termination(self)
        for k in self.learners.keys():
            if k == "gail":
                self.learners[k].save_transition(
                    self.last_dof_pos, self.dof_pos)
                if self.learners[k].check_update():
                    self.learners[k].update()

        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)
        # self._update_viewer()
        if self.historical_step > 1:
            self.dof_pos_buf = torch.cat(
                [self.dof_pos_buf[:, 1:], self.dof_pos.unsqueeze(1)], dim=1)
            self.torques_buf = torch.cat(
                [self.torques_buf[:, 1:], self.torques.unsqueeze(1)], dim=1)

        if self.get_image and self.frame_count % self.vision_update_freq == 0:
            if self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.sync_frame_time(self.sim)
                # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)
            image_vectors = torch.stack(
                [self.update_image(i) for i in range(self.num_envs)], dim=0)
            self.image_buf = torch.cat(
                [image_vectors.unsqueeze(1), self.image_buf[:, :-1]], dim=1)
        if self.get_image:
            self.obs_buf[:, self.state_obs_size:] = self.image_buf.flatten(1)
        self.compute_observations()

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        contact_forces = (
            self.contact_forces[:, self.feet_indices_in_bodies]).flatten(1)
        if self.historical_step > 1:
            dof_pos = (self.dof_pos_buf -
                       self.default_dof_pos.unsqueeze(1)).view(self.num_envs, -1)
            actions = self.actions_buf.view(self.num_envs, -1)
            self.obs_buf[:, :self.state_obs_size] = self.compute_a1_observations(  # tensors
                self.commands,
                dof_pos,
                self.dof_vel,
                self.gravity_vec,
                self.action_scale.repeat(self.historical_step) * actions,
                # scales
                self.lin_vel_scale,
                self.ang_vel_scale,
                self.dof_pos_scale,
                self.dof_vel_scale,
                contact_forces,
            )
        else:
            self.obs_buf[:, :self.state_obs_size] = self.compute_a1_observations(  # tensors
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
                contact_forces,
            )

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg["env"]["learn"]["friction_range"]
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * \
                    self.cfg["env"]["learn"]["soft_dof_pos_limit"]
                self.dof_pos_limits[i, 1] = m + 0.5 * r * \
                    self.cfg["env"]["learn"]["soft_dof_pos_limit"]
        for i in range(self.num_dof):
            props['driveMode'][i] = gymapi.DOF_MODE_POS
            props['stiffness'][i] = self.Kp
            props['damping'][i] = self.Kd
        return props

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        for _, randomizer in self.randomizer.items():
            randomizer.apply_randomizations(self)

        # positions_offset = torch_rand_float(
        #     0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.05, 0.05,
                                      (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + 0.1 * \
            (torch.rand_like(
                self.default_dof_pos[env_ids], device=self.device) - 0.5)
        self.dof_vel[env_ids] = velocities
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids]

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

        if self.historical_step > 1:
            self.dof_pos_buf[env_ids] = 0
            self.actions_buf[env_ids] = 0
            self.torques_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.feet_air_time[env_ids] = 0

    def update_image(self, env_ids):
        # output image and then write it to disk using Pillow
        # communicate physics to graphics system
        image_vec = []
        if self.image_type != "rgb":
            depth_image = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_DEPTH)
            depth_image = gymtorch.wrap_tensor(depth_image)

            # -inf implies no depth value, set it to zero. output will be black.
            depth_image[torch.isneginf(depth_image)] = 0

            # clamp depth image to 10 meters to make output image human friendly
            depth_image[depth_image < -5] = -5

            # depth_image = (depth_image - torch.mean(depth_image)) / \
            #     (torch.std(depth_image) + 1e-5)
            image_vec.append(depth_image.unsqueeze(0))

        if self.image_type != "depth":
            rgba_image = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_COLOR)
            rgb_image = gymtorch.wrap_tensor(rgba_image)[:3].float()
            rgb_image = rgba_image / 255.

            image_vec.append(rgb_image)

        image_vec = torch.cat(image_vec, dim=0).flatten()
        if self.device == "cpu":
            image_vec = image_vec.to(self.device)
        # # flip the direction so near-objects are light and far objects are dark
        # normalized_depth = -255.0 * \
        #     (depth_image / torch.min(depth_image + 1e-4))
        # normalized_depth = normalized_depth.cpu().numpy()
        # normalized_depth_image = im.fromarray(
        #     normalized_depth.astype(np.uint8), mode="L")
        # normalized_depth_image.save(
        #     "output_images/depth_{}.png".format(self.frame_count))
        # self.gym.write_camera_image_to_file(
        #     self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_COLOR, "output_images/rgb_{}.png".format(self.frame_count))
        # self.frame_count += 1
        return image_vec

    def reset_command(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.command_type == "vel":
            self.commands_x[env_ids] = torch_rand_float(
                self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
            self.commands_y[env_ids] = torch_rand_float(
                self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
            self.commands_yaw[env_ids] = torch_rand_float(
                self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        elif self.command_type == "acc":
            self.commands_x[env_ids] = - self.commands_x[env_ids]
            self.commands_y[env_ids] = - self.commands_y[env_ids]
            self.commands_yaw[env_ids] = - self.commands_yaw[env_ids]

    def _build_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None
        self.refEnv = self.env_wrapper.env_cfg["viewer"]["refEnv"]
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            p = self.env_wrapper.env_cfg["viewer"]["pos"]
            lookat = self.env_wrapper.env_cfg["viewer"]["lookat"]
            self.camera_distance = [
                _lookat - _p for _lookat, _p in zip(lookat, p)]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

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
        angles = angles.reshape(self.num_envs, self.num_legs, 3)
        foot_positions = torch.zeros_like(angles, device=self.device)
        for i in range(self.num_legs):
            foot_positions[:, i] = self._foot_position_in_hip_frame(
                angles[:, i], l_hip_sign=(-1)**(i + 1))
        return foot_positions + self._hip_offset

    def _footPositionsToJointAngles(self, foot_positions):
        fjt = self.jacobian_tensor[:, :, :, 6:]
        fjt = fjt[:, self.feet_indices + 1]
        # solve damped least squares
        fjt_T = torch.transpose(fjt, -1, -2)
        feet_pos = self._footPositionsInBaseFrame()
        feet_pos_err, feet_rot_err = torch.zeros_like(
            feet_pos), torch.zeros_like(feet_pos)
        feet_pos_err = foot_positions - feet_pos
        dof_err = torch.cat([feet_pos_err, feet_rot_err],
                            dim=-1).unsqueeze_(-1)
        d = 0.05  # damping term
        lmbda = torch.eye(6).unsqueeze_(0).unsqueeze_(0).repeat(
            self.num_envs, 4, 1, 1).to(self.device) * (d ** 2)
        u = (fjt_T @ torch.inverse(fjt @ fjt_T + lmbda) @ dof_err).squeeze(-1)
        # u = (fjt_T @ dof_err).squeeze(-1)
        delta_pos = []
        for i in range(self.num_legs):
            delta_pos.append(u[:, i, 3 * i: 3 * (i + 1)])
        delta_pos = torch.cat(delta_pos, dim=-1)
        pos_target = self.dof_pos + delta_pos
        return pos_target

    def _getContactFootState(self):
        contact_forces = self.contact_forces[:, self.feet_indices].sum(-1)
        contact_states = (contact_forces != 0)
        return contact_states

    # ============ Reward Functions ==============

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self.reward_functions = []
        self.reward_names = []

        # reward scales
        self.reward_scales = self.task_wrapper.task_cfg["learn"]["reward_scales"]
        self.reward_params = self.task_wrapper.task_cfg["learn"]["reward_params"]
        self.num_rew_terms = len(self.reward_scales)

        for key in self.reward_scales.keys():
            self.reward_scales[key] *= self.dt

        for name in self.reward_scales.keys():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        if self.task_wrapper.task_name == "following_command":
            # command ranges
            self.command_x_range = self.task_wrapper.task_cfg["randomCommandRanges"]["linear_x"]
            self.command_y_range = self.task_wrapper.task_cfg["randomCommandRanges"]["linear_y"]
            self.command_yaw_range = self.task_wrapper.task_cfg["randomCommandRanges"]["yaw"]

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(
            1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.reward_params["base_height_target"])

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = - \
            (self.dof_pos -
             self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos -
                          self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["env"]["learn"]["soft_dof_vel_limit"]).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg["env"]["learn"]["soft_torque_limit"]).clip(min=0.), dim=1)

    def _reward_moving_forward(self):
        # encourage moving forward as fast as possible
        base_lin_val_in_world_frame = self.root_states[self.a1_indices,
                                                       7] * self.lin_vel_scale
        return torch.clamp(base_lin_val_in_world_frame, min=-1000., max=self.reward_params["forward_vel"])

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_params["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_params["tracking_sigma"])

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1)
        # no reward for zero command
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.reward_params["max_contact_force"]).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1)
        # no reward for zero command
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact
        return rew_airTime * self.reward_scales["feet_air_time"]

    def _reward_gail(self):
        # GAIL reward
        return self.learners["gail"].reward(self.last_dof_pos, self.dof_pos) * self.reward_scales["gail"]

    def compute_a1_observations(self,
                                commands: Tensor,
                                dof_pos: Tensor,
                                dof_vel: Tensor,
                                gravity_vec: Tensor,
                                actions: Tensor,
                                lin_vel_scale: float,
                                ang_vel_scale: float,
                                dof_pos_scale: float,
                                dof_vel_scale: float,
                                contact_forces: Tensor
                                ) -> Tensor:

        projected_gravity = quat_rotate(self.base_quat, gravity_vec)

        commands_scaled = commands[..., :3] * \
            torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale],
                         requires_grad=False, device=commands.device)

        obs = torch.cat((
                        dof_pos * dof_pos_scale,
                        dof_vel * dof_vel_scale,
                        self.base_lin_vel,
                        self.base_ang_vel,
                        projected_gravity,
                        commands_scaled,
                        commands[..., 3:],
                        actions,
                        contact_forces,
                        ), dim=-1)

        return obs
