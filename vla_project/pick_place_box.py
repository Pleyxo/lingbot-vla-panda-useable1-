"""
Pick-and-Place into Box environment for single-arm Panda robot.

Task: Pick up the red cube from the table and place it into the open box (bin)
      located on the table next to the cube.

State:  14 dim = joint_cos(7) + eef_pos(3) + gripper_qpos(1) + cube_pos(3)
Action: 8 dim  = joint_vel(7) + gripper_cmd(1)
"""
from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_bin_arena import TableBinArena


class PickPlaceBox(SingleArmEnv):
    """
    Single-arm robot picks up a cube and places it into an open box (bin).

    The bin is positioned on the table. The cube starts on the table,
    separated from the bin. The robot must grasp the cube, move it to
    the bin, and release it inside.

    Success: cube is inside the bin (within bin walls) and NOT being grasped.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        bin_pos=(0.15, 0.15, 0.82),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.bin_pos = bin_pos
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 2.25

        elif self.reward_shaping:
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            bin_center = self._get_bin_center()

            dist_to_cube = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist_to_cube)
            reward += reaching_reward

            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25
                dist_to_bin = np.linalg.norm(cube_pos - bin_center)
                placing_reward = 1 - np.tanh(5.0 * dist_to_bin)
                reward += placing_reward

                if self._check_cube_in_bin():
                    gripper_open = self.robots[0].gripper.current_qq[-1] > 0
                    if gripper_open:
                        reward += 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableBinArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            bin_pos=self.bin_pos,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.15, -0.05],
                y_range=[-0.15, -0.05],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _get_bin_center(self):
        bin_body_id = self.sim.model.body_name2id("target_bin")
        return np.array(self.sim.data.body_xpos[bin_body_id])

    def _check_cube_in_bin(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        bin_pos = self._get_bin_center()

        bin_half_xy = 0.07
        bin_bottom_z = bin_pos[2]
        bin_top_z = bin_pos[2] + 0.08

        in_xy = (abs(cube_pos[0] - bin_pos[0]) < bin_half_xy and
                 abs(cube_pos[1] - bin_pos[1]) < bin_half_xy)
        in_z = bin_bottom_z < cube_pos[2] < bin_top_z

        return in_xy and in_z

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def bin_pos(obs_cache):
                return self._get_bin_center()

            @sensor(modality=modality)
            def cube_to_bin_pos(obs_cache):
                return obs_cache.get("cube_pos", np.zeros(3)) - self._get_bin_center()

            sensors = [cube_pos, cube_quat, gripper_to_cube_pos, bin_pos, cube_to_bin_pos]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube
            )

    def _check_success(self):
        if not self._check_cube_in_bin():
            return False
        is_grasping = self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.cube
        )
        return not is_grasping
