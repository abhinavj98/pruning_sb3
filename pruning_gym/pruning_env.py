import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, Tuple, Any, List
import os
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nptyping import NDArray, Shape, Float

from .optical_flow import OpticalFlow

from .tree import Tree
from .ur5_utils import UR5
from .pyb_utils import pyb_utils
from pruning_sb3.pruning_gym import ROBOT_URDF_PATH
from .reward_utils import Reward
from skimage.draw import disk

import copy
#TODO: Write test cases for this file
class PruningEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    # optical_flow_model = OpticalFlow()

    def __init__(self, tree_urdf_path: str, tree_obj_path: str, renders: bool = False, max_steps: int = 500,
                 learning_param: float = 0.05, tree_count: int = 9999, width: int = 224, height: int = 224,
                 evaluate: bool = False, num_points: Optional[int] = None, action_dim: int = 12,
                 name: str = "PruningEnv",
                 terminate_on_singularity: bool = True, action_scale: int = 1, movement_reward_scale: int = 1,
                 distance_reward_scale: int = 1, condition_reward_scale: int = 1, terminate_reward_scale: int = 1,
                 collision_reward_scale: int = 1, slack_reward_scale: int = 1,
                 perpendicular_orientation_reward_scale: int = 1, pointing_orientation_reward_scale: int = 1,
                 use_optical_flow: bool = False, optical_flow_subproc: bool = False,
                 shared_var: Tuple[Optional[Any], Optional[Any]] = (None, None), scale: bool = False,
                 curriculum_distances: Tuple = (0.26,), curriculum_level_steps: Tuple = (),
                 use_ik: bool = True) -> None:
        super(PruningEnv, self).__init__()

        assert tree_urdf_path is not None
        assert tree_obj_path is not None

        # TODO: Assumes shared variables is always present, to not use shared var have to pass (None, None). Make
        #  shared_var as None
        self.shared_queue = shared_var[0]
        self.shared_dict = shared_var[1]
        self.pid = os.getpid()
        # self.shared_dict[self.pid] = None
        # Pybullet GUI variables
        self.renders = renders
        self.render_mode = 'rgb_array'
        self.eval = evaluate
        # Obj/URDF paths
        self.tree_urdf_path = tree_urdf_path
        self.tree_obj_path = tree_obj_path
        # Gym variables
        self.name = name
        self.action_dim = action_dim
        self.stepCounter = 0
        self.maxSteps = max_steps
        self.tree_count = tree_count
        self.action_scale = action_scale
        self.terminated = False
        self.use_optical_flow = use_optical_flow
        self.optical_flow_subproc = optical_flow_subproc

        if self.use_optical_flow:
            if not self.optical_flow_subproc:
                self.optical_flow_model = OpticalFlow(subprocess=False)
        #     self.optical_flow_model = OpticalFlow()
        # Reward variables

        # New class for reward
        self.reward = Reward(movement_reward_scale, distance_reward_scale, pointing_orientation_reward_scale,
                             perpendicular_orientation_reward_scale, terminate_reward_scale, collision_reward_scale,
                             slack_reward_scale, condition_reward_scale)

        self.pyb = pyb_utils(self, renders, width, height)

        self.observation_space = spaces.Dict({
            'depth_proxy': spaces.Box(low=-1.,
                                      high=1.0,
                                      shape=((3, self.pyb.height, self.pyb.width) if self.use_optical_flow else (
                                          1, self.pyb.height, self.pyb.width)),
                                      dtype=np.float32),
            'desired_goal': spaces.Box(low=-5.,
                                       high=5.,
                                       shape=(3,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-5.,
                                        high=5.,
                                        shape=(3,), dtype=np.float32),
            'achieved_or': spaces.Box(low=-5.,
                                      high=5.,
                                      shape=(6,), dtype=np.float32),
            'joint_angles': spaces.Box(low=-1,
                                       high=1,
                                       shape=(12,), dtype=np.float32),
            # 'joint_velocities': spaces.Box(low=-6,
            #                                high=6,
            #                                shape=(6,), dtype=np.float32),
            'prev_action': spaces.Box(low=-1., high=1.,
                                      shape=(self.action_dim,), dtype=np.float32),
            'relative_distance': spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32),
            'critic_perpendicular_cosine_sim': spaces.Box(low=-0., high=1., shape=(1,), dtype=np.float32),
            'critic_pointing_cosine_sim': spaces.Box(low=-0., high=1., shape=(1,), dtype=np.float32),

        })
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_dim,), dtype=np.float32)

        self.sum_reward = 0.0  # Pass this to reward class
        self.scale = scale

        self.use_ik = use_ik
        self.action = np.zeros(self.action_dim)

        self.reset_counter = 0
        self.episode_counter = 0
        self.randomize_tree_count = 1
        self.learning_param = learning_param

        # setup robot arm:
        # new class for ur5
        self.ur5 = UR5(self.pyb.con, ROBOT_URDF_PATH, pos=[0.5,0,0])
        self.reset_env_variables()

        # Tree parameters
        self.tree_goal_pos = np.array([1, 0, 0])  # initial object pos
        self.tree_goal_branch = np.array([0, 0, 0])
        pos = None
        scale = None
        if "envy" in self.tree_urdf_path:
            pos = np.array([0., -0.6, 0])
            scale = 1
        elif "ufo" in self.tree_urdf_path:
            pos = np.array([-0.5, -0.8, -0.3])
            scale = 1

        assert scale is not None
        assert pos is not None
        self.trees = Tree.make_trees_from_folder(self, self.pyb, self.tree_urdf_path, self.tree_obj_path, pos=pos,
                                                 orientation=np.array([0, 0, 0, 1]), scale=scale, num_points=num_points,
                                                 num_trees=self.tree_count, curriculum_distances=curriculum_distances,
                                                 curriculum_level_steps=curriculum_level_steps)

        for tree in self.trees:
            self.tree = tree
            self.tree.active()
            tree.make_curriculum(self.ur5.init_pos[1])
            self.tree.inactive()

        self.tree = random.sample(self.trees, 1)[0]
        self.tree.active()
        for i in range(len(self.tree.curriculum_distances)):
            self.pyb.visualize_points(self.tree.curriculum_points[i])
            import time
            time.sleep(2)
            self.pyb.remove_debug_items("step")

        #Curriculum variables
        self.eval_counter = 0
        self.curriculum_level = 0
        self.curriculum_level_steps = curriculum_level_steps
        self.curriculum_distances = curriculum_distances

        # Init and final logging
        self.init_distance = 0
        self.init_point_cosine_sim = 0
        self.init_perp_cosine_sim = 0
        self.collisions_unacceptable = 0
        self.collisions_acceptable = 0

        # Observation dicts
        self.observation: dict = dict()
        self.observation_info = dict()
        self.prev_observation_info: dict = dict()

    def reset_env_variables(self) -> None:
        # Env variables that will change
        self.observation: dict = dict()
        self.observation_info = dict()
        self.prev_observation_info: dict = dict()
        self.stepCounter = 0
        self.sum_reward = 0
        self.terminated = False
        self.collisions_acceptable = 0
        self.collisions_unacceptable = 0

    def set_curriculum_level(self, episode_counter, curriculum_level_steps):
        """Set curriculum level"""
        if "eval" in self.name:
            self.curriculum_level = len(curriculum_level_steps)
        else:
            if episode_counter in curriculum_level_steps:
                self.curriculum_level += 1

    @staticmethod
    def sample_point(name, episode_counter, curriculum_points) -> Tuple[
        Tuple[float, float, float], Tuple[float, float, float]]:
        """Sample a point from the tree"""
        if "eval" in name:
            distance, random_point = curriculum_points[episode_counter % len(curriculum_points)]
            print("Eval counter: ", episode_counter, "Point: ", random_point, "points ", len(curriculum_points))
        else:
            # print(random.sample(curriculum_points, 1))
            distance, random_point = random.sample(curriculum_points, 1)[0]
        return distance, random_point

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[dict, dict]:
        """Environment reset function"""
        super().reset(seed=seed)
        random.seed(seed)
        self.reset_env_variables()
        self.reset_counter += 1
        # Remove and add tree to avoid collisions with tree while resetting
        self.tree.inactive()

        # Function for remove body
        self.ur5.remove_ur5_robot()
        self.pyb.remove_debug_items("reset")
        self.pyb.remove_debug_items("step")

        # Sample new tree if reset_counter is a multiple of randomize_tree_count
        self.set_curriculum_level(self.episode_counter, self.curriculum_level_steps)

        if self.reset_counter % self.randomize_tree_count == 0:
            while True:
                self.tree = random.sample(self.trees, 1)[0]
                if len(self.tree.curriculum_points[self.curriculum_level]) > 0:
                    break

        # Create new ur5 arm body
        self.ur5.setup_ur5_arm()  # Remember to remove previous body! Line 215

        # Make this a new function that supports curriculum
        # Set curriculum level

        # Sample new point
        distance_from_goal, random_point = self.sample_point(self.name, self.episode_counter,
                                                             self.tree.curriculum_points[self.curriculum_level])
        print("Distance from goal: ", distance_from_goal, "Point: ", random_point)
        if "record" in self.name:
            # Move to pybullet class
            """Display red sphere during evaluation"""
            self.pyb.add_debug_item('sphere', 'reset', lineFromXYZ=random_point[0],
                                    lineToXYZ=[random_point[0][0]+0.005, random_point[0][1]+0.005,\
                                               random_point[0][2]+0.005],
                                    lineColorRGB=[1, 0, 0],
                                    lineWidth=200)
        # here the arm is set right in front of the point -> Change this to curriculum
        # self.set_joint_angles(
        #    self.calculate_ik((random_point[0][0] , random_point[0][1] + distance_from_goal, random_point[0][2]), self.ur5.init_pos[1]))
        self.ur5.set_joint_angles(
            self.ur5.calculate_ik((self.ur5.init_pos[0][0], self.ur5.init_pos[0][1] + 0.05, self.ur5.init_pos[0][2]),
                                  self.ur5.init_pos[1]))

        for i in range(100):
            self.pyb.con.stepSimulation()
        self.tree_goal_pos = random_point[0]
        self.tree_goal_branch = random_point[1]
        self.tree.active()

        pos, orient = self.ur5.get_current_pose(self.ur5.end_effector_index)

        # Logging
        self.init_distance = np.linalg.norm(self.tree_goal_pos - self.ur5.init_pos[0]) + 1e-4
        self.init_point_cosine_sim = self.reward.compute_pointing_cos_sim(np.array(pos), self.tree_goal_pos,
                                                                          np.array(orient),
                                                                          self.tree_goal_branch) + 1e-4
        self.init_perp_cosine_sim = self.reward.compute_perpendicular_cos_sim(np.array(orient),
                                                                              self.tree_goal_branch) + 1e-4

        # Add debug branch
        _ = self.pyb.add_debug_item('line', 'step', lineFromXYZ=self.tree_goal_pos - 50 * self.tree_goal_branch,
                                    lineToXYZ=self.tree_goal_pos + 50 * self.tree_goal_branch, lineColorRGB=[1, 0, 0],
                                    lineWidth=200)
        self.set_extended_observation()
        info = dict()  # type: ignore
        # Make info analogous to one in step function
        return self.observation, info


    def calculate_joint_velocities_from_ee_constrained(self, velocity):
        """Calculate joint velocities from end effector velocities adds constraints of max joint velocities"""
        if self.use_ik:
            jv, jacobian = self.ur5.calculate_joint_velocities_from_ee_velocity(velocity)
            # check if actual ee velocity is close to desired ee velocity
            actual_ee_vel = np.matmul(jacobian, jv)
            #TODO:Log ee_vel_error
            self.ee_vel_error = abs(actual_ee_vel - velocity) / (velocity + 1e-5)
            if (self.ee_vel_error > 0.1).any():
                print("Nope")
                jv = np.zeros(6)
        #If not using ik, just set joint velocities to be regressed by actor
        else:
            jv = velocity
        return jv
    def step(self, action: NDArray[Shape['6, 1'], Float]) -> Tuple[dict, float, bool, bool, dict]:
        self.pyb.remove_debug_items("step")
        self.action = action

        action = action * self.action_scale
        self.ur5.action = self.calculate_joint_velocities_from_ee_constrained(action)
        singularity = self.ur5.set_joint_velocities(self.ur5.action)

        for i in range(1):
            self.pyb.con.stepSimulation()
            # print(self.pyb.con.getJointStateMultiDof(self.ur5.ur5_robot, self.ur5.end_effector_index))
            # if self.renders: time.sleep(5./240.) 

        # Need observations before reward
        self.set_extended_observation()
        current_pose = np.hstack((self.observation_info['achieved_pos'], self.observation_info['achieved_or']))
        if 'achieved_pos' not in self.prev_observation_info.keys():
            self.prev_observation_info['achieved_pos'] = np.zeros(3)
            self.prev_observation_info['achieved_or'] = np.zeros(4)
        previous_pose = np.hstack(
            (self.prev_observation_info['achieved_pos'], self.prev_observation_info['achieved_or']))
        reward, reward_infos = self.compute_reward(self.observation_info['desired_pos'], current_pose,
                                                   previous_pose, singularity,
                                                   None)

        self.sum_reward += reward
        # self.debug_line = self.pyb.con.addUserDebugLine(self.achieved_pos, self.desired_pos, [0, 0, 1], 20)
        self.stepCounter += 1

        done, terminate_info = self.is_task_done()
        truncated = terminate_info['time_limit_exceeded']
        terminated = terminate_info['goal_achieved']
        infos = {'is_success': False, "TimeLimit.truncated": False}  # type: ignore
        if terminate_info['time_limit_exceeded']:
            infos["TimeLimit.truncated"] = True  # type: ignore
            infos["terminal_observation"] = self.observation  # type: ignore

        if self.terminated is True:
            infos['is_success'] = True

        # Logging end of episode info
        if truncated or terminated:
            self.episode_counter += 1
            infos['episode'] = {"l": self.stepCounter, "r": self.sum_reward}  # type: ignore
            print("Episode Length: ", self.stepCounter)
            infos["pointing_cosine_sim_error"] = Reward.compute_pointing_cos_sim(
                achieved_pos=self.observation_info['achieved_pos'],
                desired_pos=self.observation_info['desired_pos'],
                achieved_or=self.observation_info['achieved_or'],
                branch_vector=self.tree_goal_branch)
            infos["perpendicular_cosine_sim_error"] = Reward.compute_perpendicular_cos_sim(
                achieved_or=self.observation_info['achieved_or'], branch_vector=self.tree_goal_branch)
            infos["euclidean_error"] = np.linalg.norm(
                self.observation_info['achieved_pos'] - self.observation_info['desired_pos'])

        # infos['episode'] = {"l": self.stepCounter,  "r": reward}
        infos['velocity'] = np.linalg.norm(self.action)
        infos.update(reward_infos)
        # return self.observation, reward, done, infos
        # v26
        return self.observation, reward, terminated, truncated, infos

    def render(self, mode="rgb_array") -> NDArray:  # type: ignore
        img_rgbd = self.pyb.get_image_at_curr_pose(type='viz')
        img_rgb, _ = self.pyb.seperate_rgbd_rgb_d(img_rgbd, 512, 512)
        if mode == "human":
            import cv2
            cv2.imshow("img", (img_rgb * 255).astype(np.uint8))
            cv2.waitKey(1)

        return img_rgb

    def close(self) -> None:
        self.pyb.con.disconnect()

    def compute_deprojected_point_mask(self):
        # TODO: Make this function nicer
        # Function. Be Nice.

        point_mask = np.zeros((self.pyb.height, self.pyb.width), dtype=np.float32)
        point_mask = np.expand_dims(point_mask, axis=0).astype(np.float32)
        proj_matrix = np.asarray(self.pyb.proj_mat).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.ur5.get_view_mat_at_curr_pose()).reshape([4, 4], order="F")
        projection = proj_matrix @ view_matrix @ np.array(
            [self.tree_goal_pos[0], self.tree_goal_pos[1], self.tree_goal_pos[2], 1])
        # Normalize by w
        projection = projection / projection[3]

        # if projection within 1,-1, set point mask to 1
        if projection[0] < 1 and projection[0] > -1 and projection[1] < 1 and projection[1] > -1:
            projection = (projection + 1) / 2
            row = self.pyb.height - 1 - int(projection[1] * (self.pyb.height))
            col = int(projection[0] * self.pyb.width)
            radius = 5  # TODO: Make this a variable proportional to distance
            # modern scikit uses a tuple for center
            rr, cc = disk((row, col), radius)
            point_mask[0, np.clip(0, rr, 223), np.clip(0, cc,
                                                       223)] = 1  # TODO: This is a hack, numbers shouldnt exceed max and min anyways

        return point_mask

    def get_depth_proxy(self, use_optical_flow, optical_flow_subproc, prev_rgb):
        rgb, depth = self.pyb.get_rgbd_at_cur_pose('robot', self.ur5.get_view_mat_at_curr_pose())
        point_mask = self.compute_deprojected_point_mask()
        if use_optical_flow:
            # if subprocvenv add the rgb to the queue and wait for the optical flow to be calculated
            if optical_flow_subproc:
                self.shared_queue.put((rgb, prev_rgb, self.pid))
                while not self.pid in self.shared_dict.keys():
                    pass
                optical_flow = self.shared_dict[self.pid]
                depth_proxy = np.concatenate((optical_flow, point_mask))
                del self.shared_dict[self.pid]
            else:
                optical_flow = self.optical_flow_model.calculate_optical_flow(rgb, prev_rgb)
                depth_proxy = np.concatenate((optical_flow, point_mask))
        else:
            depth_proxy = np.expand_dims(depth.astype(np.float32), axis=0)
            depth_proxy = np.concatenate((depth_proxy, point_mask))
        return depth_proxy, rgb

    def set_extended_observation(self) -> dict:
        """
        The observations are the current position, the goal position, the current orientation, the current depth
        image, the current joint angles and the current joint velocities
        """
        #TODO: define all these dict as named tuples/dict
        self.prev_observation_info = copy.deepcopy(self.observation_info)

        tool_pos, tool_orient = self.ur5.get_current_pose(self.ur5.end_effector_index)
        achieved_vel, achieved_ang_vel = self.ur5.get_current_vel(self.ur5.end_effector_index)

        achieved_pos = np.array(tool_pos).astype(np.float32)
        achieved_or = np.array(tool_orient).astype(np.float32)
        desired_pos = np.array(self.tree_goal_pos).astype(np.float32)

        joint_angles = np.array(self.ur5.get_joint_angles()).astype(np.float32)
        init_pos = np.array(self.ur5.init_pos[0]).astype(np.float32)
        init_or = np.array(self.ur5.init_pos[1]).astype(np.float32)

        achieved_or_mat = np.array(self.pyb.con.getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        achieved_or_6d = achieved_or_mat[:, :2].reshape(6, ).astype(np.float32)

        if 'rgb' not in self.prev_observation_info:
            self.prev_observation_info['rgb'] = np.zeros((self.pyb.height, self.pyb.width, 3))
        depth_proxy, rgb = self.get_depth_proxy(self.use_optical_flow, self.optical_flow_subproc,
                                                prev_rgb=self.prev_observation_info[
                                                    'rgb'], )  # Get this from previous obs
        encoded_joint_angles = np.hstack((np.sin(joint_angles), np.cos(joint_angles)))

        pointing_cosine_sim = self.reward.compute_pointing_cos_sim(achieved_pos, desired_pos, achieved_or,
                                                                   self.tree_goal_branch)
        perpendicular_cosine_sim = self.reward.compute_perpendicular_cos_sim(achieved_or, self.tree_goal_branch)

        # Just infos to be used in the reward function/other methods
        self.observation_info['desired_pos'] = desired_pos
        self.observation_info['achieved_pos'] = achieved_pos
        self.observation_info['achieved_or'] = achieved_or
        self.observation_info['rgb'] = rgb
        self.observation_info['pointing_cosine_sim'] = pointing_cosine_sim
        self.observation_info['perpendicular_cosine_sim'] = perpendicular_cosine_sim
        self.observation_info['target_distance'] = np.linalg.norm(achieved_pos - desired_pos)

        # Actual observation
        self.observation['achieved_goal'] = achieved_pos - init_pos
        self.observation['desired_goal'] = desired_pos - init_pos
        self.observation['relative_distance'] = achieved_pos - desired_pos
        # Convert orientation into 6D form for continuity
        self.observation['achieved_or'] = achieved_or_6d
        self.observation['depth_proxy'] = depth_proxy
        # Convert joint angles to sin and cos
        self.observation['joint_angles'] = encoded_joint_angles
        # self.observation[
        #     'joint_velocities'] = self.ur5.action  # Check the name of this variable and figure where it is set
        # Action actually achieved

        self.observation['prev_action'] = np.hstack((achieved_vel, achieved_ang_vel))

        # Privileged critic
        # Add cosine sim perp and point
        self.observation['critic_pointing_cosine_sim'] = np.array(pointing_cosine_sim).astype(np.float32).reshape(1, )
        self.observation['critic_perpendicular_cosine_sim'] = np.array(perpendicular_cosine_sim).astype(
            np.float32).reshape(1, )

        return self.observation

    def is_task_done(self) -> Tuple[bool, dict]:
        # NOTE: need to call compute_reward before this to check termination!
        time_limit_exceeded = self.stepCounter >= self.maxSteps
        goal_achieved = self.terminated
        c = (self.terminated is True or self.stepCounter > self.maxSteps)
        terminate_info = {"time_limit_exceeded": time_limit_exceeded,
                          "goal_achieved": goal_achieved}
        return c, terminate_info

    def compute_reward(self, desired_goal: NDArray[Shape['3, 1'], Float], achieved_pose: NDArray[Shape['6, 1'], Float],
                       previous_pose: NDArray[Shape['6, 1'], Float], singularity: bool, info: dict) -> Tuple[
        float, dict]:

        achieved_pos = achieved_pose[:3]
        achieved_or = achieved_pose[3:]
        desired_pos = desired_goal
        previous_pos = previous_pose[:3]
        previous_or = previous_pose[3:]
        reward = 0.0
        # There will be two different types of achieved positions, one for the end effector and one for the camera

        self.collisions_acceptable = 0
        self.collisions_unacceptable = 0
        _ = self.pyb.add_debug_item('line', 'step', lineFromXYZ=achieved_pos, lineToXYZ=desired_pos,
                                    lineColorRGB=[0, 0, 1], lineWidth=20)
        _ = self.pyb.add_debug_item('line', 'step', lineFromXYZ=previous_pos, lineToXYZ=desired_pos,
                                    lineColorRGB=[0, 0, 1], lineWidth=20)

        # Calculate rewards
        reward += self.reward.calculate_distance_reward(achieved_pos, desired_pos)
        reward += self.reward.calculate_movement_reward(achieved_pos, previous_pos, desired_pos)
        point_reward, point_cosine_sim = self.reward.calculate_pointing_orientation_reward(achieved_pos, desired_pos,
                                                                                           achieved_or, previous_pos,
                                                                                           previous_or,
                                                                                           self.tree_goal_branch)
        reward += point_reward

        perp_reward, perp_cosine_sim = self.reward.calculate_perpendicular_orientation_reward(achieved_or, previous_or,
                                                                                              self.tree_goal_branch)
        reward += perp_reward

        condition_number = self.ur5.get_condition_number()
        reward += self.reward.calculate_condition_number_reward(condition_number)

        # If is successful
        self.terminated = self.is_state_successful(achieved_pos, desired_pos, perp_cosine_sim, point_cosine_sim)
        reward += self.reward.calculate_termination_reward(self.terminated)

        is_collision, collision_info = self.ur5.check_collisions(self.tree.tree_id, self.tree.supports)
        reward += self.reward.calculate_acceptable_collision_reward(collision_info)
        reward += self.reward.calculate_unacceptable_collision_reward(collision_info)

        # check collisions:
        if is_collision:
            self.log_collision_info(collision_info)

        reward += self.reward.calculate_slack_reward()
        reward += self.reward.calculate_velocity_minimization_reward(self.ur5.action)

        return reward, self.reward.reward_info

    def is_state_successful(self, achieved_pos, desired_pos, orientation_perp_value, orientation_point_value):
        terminated = False
        is_success_collision = self.ur5.check_success_collision(self.tree.tree_id)
        dist_from_target = np.linalg.norm(achieved_pos - desired_pos)
        if is_success_collision and dist_from_target < self.learning_param:
            if (orientation_perp_value > 0.7) and (
                    orientation_point_value > 0.7):  # and approach_velocity < 0.05:
                terminated = True

        return terminated

    def log_collision_info(self, collision_info):
        if collision_info['collisions_acceptable']:
            self.collisions_acceptable += 1
            # print('Collision acceptable!')
        elif collision_info['collisions_unacceptable']:
            self.collisions_unacceptable += 1
            # print('Collision unacceptable!')
