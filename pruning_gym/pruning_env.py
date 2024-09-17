import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from memory_profiler import profile
from typing import Optional, Tuple
import random
from pruning_sb3.pruning_gym import label
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nptyping import NDArray, Shape, Float
import imageio
import pickle
from .tree import Tree
from .ur5_utils import UR5
from .pyb_utils import pyb_utils
from .reward_utils import Reward
from skimage.draw import disk
from pruning_sb3.pruning_gym import ROBOT_URDF_PATH, SUPPORT_AND_POST_PATH
import copy
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import pybullet_planning as pp
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_sample_fn, get_extend_fn, \
    get_distance_fn, get_difference_fn
from pybullet_planning.interfaces.robots import get_collision_fn
from pybullet_planning.motion_planners.smoothing import smooth_path, refine_waypoints
from pruning_sb3.baselines.rrt_star_with_tolerance import informed_rrt_star_multi_goal, rrt_star_multi_goal
import time
import pandas as pd
from enum import Enum
from scipy.stats import vonmises
from filelock import FileLock
class PruningEnv(gym.Env):
    """
        PruningEnv is a custom environment that extends the gym.Env class from OpenAI Gym.
        This environment simulates a pruning task where a robot arm interacts with a tree.
        The robot arm is a UR5 arm and the tree is a 3D model of a tree.
        The environment is used to train a reinforcement learning agent to prune the tree.
    """

    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, tree_urdf_path: str, tree_obj_path: str, tree_labelled_path: str, renders: bool = False,
                 max_steps: int = 500,
                 distance_threshold: float = 0.05, angle_threshold_perp: float = 0.6,
                 angle_threshold_point: float = 0.6,
                 tree_count: int = 9999, cam_width: int = 424, cam_height: int = 240,
                 algo_width: int = 224, algo_height: int = 224,
                 evaluate: bool = False, num_points: Optional[int] = None, action_dim: int = 12,
                 name: str = "PruningEnv", action_scale: int = 1, movement_reward_scale: int = 1,
                 distance_reward_scale: int = 1, condition_reward_scale: int = 1, terminate_reward_scale: int = 1,
                 collision_reward_scale: int = 1, slack_reward_scale: int = 1,
                 perpendicular_orientation_reward_scale: int = 1, pointing_orientation_reward_scale: int = 1,
                 curriculum_distances: Tuple = (0.8,), curriculum_level_steps: Tuple = (),
                 use_ik: bool = True, make_trees: bool = False,
                 ur5_pos=[0, 0, 0], ur5_or=[0, 0, 0, 1], randomize_ur5_pose: bool = False,
                 randomize_tree_pose: bool = False,
                 verbose=1
                 ) -> None:
        """
        Initializes the environment with the following parameters:
        :param tree_urdf_path: Path to the folder containing URDF files of the trees
        :param tree_obj_path: Path to the folder containing OBJ files of the trees with UV coordinates and mtl files
        :param tree_labelled_path: Path to the folder containing labelled OBJ files of the trees - Directly from Lpy
        :param renders: Whether to render the environment
        :param max_steps: Maximum number of steps in an episode
        :param distance_threshold: Distance threshold for termination
        :param angle_threshold_perp: Angle threshold perpendicular for termination
        :param angle_threshold_point: Angle threshold pointing for termination
        :param tree_count: Number of trees to be loaded
        :param cam_width: Width of the camera
        :param cam_height: Height of the camera
        :param algo_width: Required width for optical flow
        :param algo_height: Required height for optical flow
        :param evaluate: Is this environment for evaluation
        :param num_points: Number of points to be sampled from the tree
        :param action_dim: Dimension of the action space
        :param name: Name of the environment
        :param action_scale: Scale of the action
        :param movement_reward_scale: Scale of the movement reward
        :param distance_reward_scale: Scale of the distance reward
        :param condition_reward_scale: Scale of the condition number reward
        :param terminate_reward_scale: Scale of the termination reward
        :param collision_reward_scale: Scale of the collision reward
        :param slack_reward_scale: Scale of the slack reward
        :param perpendicular_orientation_reward_scale: Scale of the perpendicular orientation reward
        :param pointing_orientation_reward_scale: Scale of the pointing orientation reward
        :param curriculum_distances: Distances for the curriculum
        :param curriculum_level_steps: Steps at which to change the curriculum
        :param use_ik: Whether to use IK for the robot arm or joint velocities
        :param make_trees: Whether to make trees from the URDF and OBJ files or load them from shared memory
        :param shared_tree_list: List of trees to be loaded from shared memory
        """

        super(PruningEnv, self).__init__()

        assert tree_urdf_path is not None
        assert tree_obj_path is not None
        assert tree_labelled_path is not None

        # Pybullet GUI variables
        self.render_mode = "rgb_array"
        self.renders = renders
        self.eval = evaluate
        # Obj/URDF paths
        self.tree_urdf_path = tree_urdf_path
        self.tree_obj_path = tree_obj_path
        self.tree_labelled_path = tree_labelled_path
        self.tree_id = None
        # Gym variables
        self.name = name
        self.action_dim = action_dim
        self.step_counter = 0
        self.global_step_counter = 0
        self.maxSteps = max_steps
        self.tree_count = tree_count
        self.action_scale = action_scale
        self.is_goal_state = False
        self.algo_width = algo_width
        self.algo_height = algo_height
        # Camera params
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_pan = 0
        self.cam_tilt = 0
        self.cam_xyz_offset = np.zeros(3)
        self.verbose = verbose
        self.collision_object_ids = {'SPUR': None, 'TRUNK': None, 'BRANCH': None, 'WATER_BRANCH': None,
                                     'SUPPORT': None, }

        # Reward variables
        self.reward = Reward(movement_reward_scale, distance_reward_scale, pointing_orientation_reward_scale,
                             perpendicular_orientation_reward_scale, terminate_reward_scale, collision_reward_scale,
                             slack_reward_scale, condition_reward_scale)

        self.pyb = pyb_utils(self, renders=renders, cam_height=cam_height, cam_width=cam_width)

        self.observation_space = spaces.Dict({
            # rgb is hwc but pytorch is chw
            'rgb': spaces.Box(low=0,
                              high=255,
                              shape=(self.cam_height, self.cam_width, 3),
                              dtype=np.uint8),
            'prev_rgb': spaces.Box(low=0,
                                   high=255,
                                   shape=(self.cam_height, self.cam_width, 3),
                                   dtype=np.uint8),
            'point_mask': spaces.Box(low=0.,
                                     high=1.,
                                     shape=(1, self.algo_height, self.algo_width),
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
            'prev_action_achieved': spaces.Box(low=-1., high=1.,
                                               shape=(self.action_dim,), dtype=np.float32),
            'relative_distance': spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32),
            'critic_perpendicular_cosine_sim': spaces.Box(low=-0., high=1., shape=(1,), dtype=np.float32),
            'critic_pointing_cosine_sim': spaces.Box(low=-0., high=1., shape=(1,), dtype=np.float32),

        })
        self.control_time = 1. / 2
        self.num_control_simulation_steps = int(self.control_time / self.pyb.step_time)  # 2Hz
        if self.verbose > 0:
            print("INFO: Control freq is ", self.control_time)
            print("INFO: Num control simulation steps is ", self.num_control_simulation_steps)
        self.label = label
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_dim,), dtype=np.float32)
        self.sum_reward = 0.0  # Pass this to reward class

        self.use_ik = use_ik
        self.action = np.zeros(self.action_dim)

        self.reset_counter = -1  # Forces random tree selection at start
        self.episode_counter = 0
        self.randomize_tree_count = 5
        self.distance_threshold = distance_threshold
        self.angle_threshold_perp = angle_threshold_perp
        self.angle_threshold_point = angle_threshold_point
        self.angle_threshold_perp_cosine = np.cos(angle_threshold_perp)
        self.angle_threshold_point_cosine = np.cos(angle_threshold_point)

        # setup robot arm:
        self.ur5 = UR5(self.pyb.con, ROBOT_URDF_PATH, pos=ur5_pos, orientation=ur5_or,
                       randomize_pose=randomize_ur5_pose, verbose=verbose)
        self.reset_env_variables()
        # Curriculum variables
        self.eval_counter = 0
        self.curriculum_level = 0
        self.curriculum_level_steps = curriculum_level_steps
        self.curriculum_distances = curriculum_distances

        # Tree parameters
        self.tree_goal_pos = None
        self.tree_goal_or = None
        self.tree_goal_normal = None
        self.tree_urdf = None
        self.tree_pos = None
        self.tree_orientation = None
        self.tree_scale = None

        pos = None
        scale = None
        if "envy" in self.tree_urdf_path:
            pos = np.array([0., -0.9, 0])
            scale = 1
        elif "ufo" in self.tree_urdf_path:
            pos = np.array([-0.5, -0.8, -0.3])
            scale = 1

        assert scale is not None
        assert pos is not None
        self.make_trees = make_trees

        if self.make_trees:
            self.trees = Tree.make_trees_from_folder(self, self.pyb, self.tree_urdf_path, self.tree_obj_path,
                                                     self.tree_labelled_path, pos=pos,
                                                     orientation=np.array([0, 0, 0, 1]), scale=scale,
                                                     num_points=num_points,
                                                     num_trees=self.tree_count,
                                                     curriculum_distances=curriculum_distances,
                                                     curriculum_level_steps=curriculum_level_steps,
                                                     randomize_pose=randomize_tree_pose)

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

    def set_tree_properties(self, tree_urdf, point_pos, point_branch_or, tree_orientation, tree_scale, tree_pos,
                            point_branch_normal):

        self.tree_urdf = tree_urdf
        self.tree_goal_pos = point_pos
        self.tree_goal_or = point_branch_or
        self.tree_pos = tree_pos
        self.tree_orientation = tree_orientation
        self.tree_scale = tree_scale
        self.tree_goal_normal = point_branch_normal
        if self.tree_id is not None:
            self.inactivate_tree(self.pyb)

        # Remove collision objects
        # Make collision shape centered at tree_goal_pos and tree_goal_normal
        # and of size 0.1
        # point_branch_or_quat = self.pyb.con.getQuaternionFromEuler(point_branch_or)
        # a = self.pyb.con.createCollisionShape(self.pyb.con.GEOM_CYLINDER, radius=.025, collisionFramePosition=point_pos,collisionFrameOrientation=point_branch_or_quat)
        # b = self.pyb.con.createVisualShape(self.pyb.con.GEOM_CYLINDER, radius=.025, visualFramePosition=point_pos, visualFrameOrientation = point_branch_or_quat, rgbaColor=[1,1,1,1])
        # self.pyb.con.createMultiBody(0, a, b)
        self.pyb.add_debug_item('sphere', 'reset', lineFromXYZ=self.tree_goal_pos,
                                lineToXYZ=[self.tree_goal_pos[0] + 0.005, self.tree_goal_pos[1] + 0.005,
                                           self.tree_goal_pos[2] + 0.005],
                                lineColorRGB=[1, 0, 0],
                                lineWidth=400)
        self.pyb.add_debug_item('line', 'reset', lineFromXYZ=self.tree_goal_pos - 50 * self.tree_goal_or,
                                    lineToXYZ=self.tree_goal_pos + 50 * self.tree_goal_or, lineColorRGB=[1, 0, 0],
                                    lineWidth=400)

        self.activate_tree(self.pyb)

    def reset_env_variables(self) -> None:
        # Env variables that will change
        self.observation: dict = dict()
        self.observation_info = dict()
        self.prev_observation_info: dict = dict()
        self.step_counter = 0
        self.sum_reward = 0
        self.is_goal_state = False
        self.collisions_acceptable = 0
        self.collisions_unacceptable = 0
        self.ur5_robot = None

    def set_curriculum_level(self, episode_counter, curriculum_level_steps):
        """Set curriculum level"""
        if "test" in self.name or self.eval:  # Replace naming with eval variable
            self.curriculum_level = len(curriculum_level_steps)
        else:
            if episode_counter in curriculum_level_steps:
                self.curriculum_level += 1
                print("Curriculum level {} at {}".format(self.curriculum_level, self.global_step_counter))

    def set_camera_pose(self):
        pan_bounds = (-2, 2)
        tilt_bounds = (-2, 2)
        self.cam_pan = np.radians(np.random.uniform(*pan_bounds))
        self.cam_tilt = np.radians(2 + np.random.uniform(*tilt_bounds))
        self.cam_xyz_offset = np.random.uniform(-1, 1, 3) * np.array([0.01, 0.0005, 0.01])

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[dict, dict]:
        """Theres a chance that this is causing some memory leak. Shows up when using reset multiple times and parallelized."""
        """Environment reset function"""
        super().reset(seed=seed)
        self.pyb.con.resetSimulation()
        self.ur5_robot = None
        self.tree_id = None
        self.collision_object_ids = {'SPUR': None, 'TRUNK': None, 'BRANCH': None, 'WATER_BRANCH': None,
                                     'SUPPORT': None, }

        # #enable file caching
        self.pyb.con.setPhysicsEngineParameter(enableFileCaching=1)

        if self.verbose > 1:
            print("DEBUG: Resetting environment")
        random.seed(seed)
        self.reset_env_variables()

        # Remove and add tree to avoid collisions with tree while resetting
        # if self.tree_id is not None:
        #     self.inactivate_tree(self.pyb)

        # Function for remove body
        # self.ur5.remove_ur5_robot()
        self.pyb.remove_debug_items("reset")
        self.pyb.remove_debug_items("step")
        # self.pyb.con.resetSimulation()

        self.set_curriculum_level(self.episode_counter, self.curriculum_level_steps)  # This will not work now

        # Create new ur5 arm body
        self.pyb.create_background()
        self.ur5.setup_ur5_arm()  # Remember to remove previous body! Line 215
        # self.ur5.reset_ur5_arm()
        # Sample new point
        # Jitter the camera pose
        self.set_camera_pose()

        for i in range(2):
            self.pyb.con.stepSimulation()

        self.activate_tree(self.pyb)
        self.pyb.add_debug_item('sphere', 'reset', lineFromXYZ=self.tree_goal_pos,
                                lineToXYZ=[self.tree_goal_pos[0] + 0.005, self.tree_goal_pos[1] + 0.005,
                                           self.tree_goal_pos[2] + 0.005],
                                lineColorRGB=[1, 0, 0],
                                lineWidth=400)
        self.pyb.add_debug_item('line', 'reset', lineFromXYZ=self.tree_goal_pos - 50 * self.tree_goal_or,
                                lineToXYZ=self.tree_goal_pos + 50 * self.tree_goal_or, lineColorRGB=[1, 0, 0],
                                lineWidth=400)

        pos, orient = self.ur5.get_current_pose(self.ur5.end_effector_index)

        # Logging
        self.init_distance = np.linalg.norm(self.tree_goal_pos - self.ur5.init_pos_ee[0]) + 1e-4
        self.init_point_cosine_sim = self.reward.compute_pointing_cos_sim(np.array(pos), self.tree_goal_pos,
                                                                          np.array(orient),
                                                                          self.tree_goal_or) + 1e-4
        self.init_perp_cosine_sim = self.reward.compute_perpendicular_cos_sim(np.array(orient),
                                                                              self.tree_goal_or) + 1e-4

        # TODO: Can we remove this?
        current_or_mat = np.array(self.pyb.con.getMatrixFromQuaternion(orient)).reshape(3, 3)
        theta, _ = Reward.get_angular_distance_to_goal(current_or_mat.T, self.tree_goal_or,
                                                       pos, self.tree_goal_pos)

        self.init_angular_error = theta

        # Add debug branch and point
        """Display red line as point"""

        self.set_extended_observation()
        info = dict()  # type: ignore
        self.reset_counter += 1
        # Make info analogous to one in step function
        self.start_time = time.time()
        return self.observation, info

    def calculate_joint_velocities_from_ee_constrained(self, velocity):
        """Calculate joint velocities from end effector velocities adds constraints of max joint velocities"""
        if self.use_ik:
            jv, jacobian = self.ur5.calculate_joint_velocities_from_ee_velocity_dls(velocity)
        # If not using ik, just set joint velocities to be regressed by actor
        else:
            jv = velocity
        return jv


    #Instead of deleting move all trees to the corner
    #Make a mapping from urdf to tree id and use that
    def activate_tree(self, pyb):
        if self.verbose > 1:
            print("DEBUG: Activating tree")
            print('DEBUG: Loading tree from ', self.tree_urdf)

        supports = pyb.con.loadURDF(SUPPORT_AND_POST_PATH, [self.tree_pos[0], self.tree_pos[1] - 0.05, 0.0],
                                    list(pyb.con.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])),
                                    globalScaling=1)

        self.collision_object_ids['SUPPORT'] = supports
        self.tree_id = pyb.con.loadURDF(self.tree_urdf, self.tree_pos, self.tree_orientation,
                                        globalScaling=self.tree_scale)
        for i in self.label.values():

            label_urdf_path = os.path.join(os.path.dirname(self.tree_urdf) + '_labelled_split',
                                           os.path.basename(self.tree_urdf).split(".")[0] + '_' + f"{i}.urdf")
            if self.verbose > 1:
                print('DEBUG: Loading tree from ', label_urdf_path)
            self.collision_object_ids[i] = pyb.con.loadURDF(label_urdf_path, self.tree_pos, self.tree_orientation,
                                                            globalScaling=self.tree_scale)

        # Do not make the textured tree collide with the UR5, let the collision objects do
        for i in range(self.ur5.num_joints):
            pyb.con.setCollisionFilterPair(self.tree_id, self.ur5.ur5_robot, 0, i, 0)

    def inactivate_tree(self, pyb):
        if self.tree_id:
            pyb.con.removeBody(self.tree_id)
            for i in self.collision_object_ids.values():
                pyb.con.removeBody(i)
            self.tree_id = None
            self.collision_object_ids = {'SPUR': None, 'TRUNK': None, 'BRANCH': None, 'WATER_BRANCH': None,
                                         'SUPPORT': None, }

    def force_time_limit(self):
        """Force time limit"""
        self.step_counter = self.maxSteps + 1

    def get_infos(self, terminated, truncated):
        infos = {'is_success': False, "TimeLimit.truncated": False}  # type: ignore
        if truncated:
            infos["TimeLimit.truncated"] = False  # type: ignore
            infos["terminal_observation"] = self.observation  # type: ignore
        if truncated or terminated:
            if self.is_goal_state is True:
                infos['is_success'] = True

            self.episode_counter += 1
            # For logging
            infos['episode'] = {"l": self.step_counter, "r": self.sum_reward}  # type: ignore
            if self.verbose > 0:
                print("INFO: Episode Length: ", self.step_counter)

            # Logging errors at the end of episode
            infos["pointing_cosine_sim_error"] = np.abs(Reward.compute_pointing_cos_sim(
                achieved_pos=self.observation_info['achieved_eebase_pos'],
                desired_pos=self.observation_info['desired_pos'],
                achieved_or=self.observation_info['achieved_eebase_or_quat'],
                branch_vector=self.tree_goal_or))

            infos["perpendicular_cosine_sim_error"] = np.abs(Reward.compute_perpendicular_cos_sim(
                achieved_or=self.observation_info['achieved_eebase_or_quat'], branch_vector=self.tree_goal_or))

            infos["euclidean_error"] = np.linalg.norm(
                self.observation_info['achieved_pos'] - self.observation_info['desired_pos'])

            current_or_mat = np.array(
                self.pyb.con.getMatrixFromQuaternion(self.observation_info['achieved_or_quat'])).reshape(3, 3)
            theta, rf = Reward.get_angular_distance_to_goal(current_or_mat.T, self.tree_goal_or,
                                                            self.observation_info['achieved_pos'],
                                                            self.observation_info['desired_pos'])
            infos["angular_error"] = theta
            infos['velocity'] = np.linalg.norm(self.action)
            infos['time'] = time.time() - self.start_time

        return infos

    def convert_local_action_to_global(self, action):
        """Convert local action to global action"""
        pos, orient = self.ur5.get_current_pose(self.ur5.end_effector_index)
        current_or_mat = np.array(self.pyb.con.getMatrixFromQuaternion(orient)).reshape(3, 3)
        global_velocity = np.dot(current_or_mat, action[:3])
        global_angular_velocity = np.dot(current_or_mat, action[3:])
        return np.hstack((global_velocity, global_angular_velocity))

    def convert_global_action_to_local(self, action):
        """Convert global action to local action"""
        pos, orient = self.ur5.get_current_pose(self.ur5.end_effector_index)
        current_or_mat = np.array(self.pyb.con.getMatrixFromQuaternion(orient)).reshape(3, 3)
        local_velocity = np.dot(current_or_mat.T, action[:3])
        local_angular_velocity = np.dot(current_or_mat.T, action[3:])
        return np.hstack((local_velocity, local_angular_velocity))



    def step(self, action: NDArray[Shape['6, 1'], Float]) -> Tuple[dict, float, bool, bool, dict]:

        self.pyb.remove_debug_items("step")
        # Scale all the actions TODO: Make scaling for rotation and translation different
        action[:3] = action[:3] * self.action_scale
        action[3:] = action[3:] * self.action_scale
        self.action = self.convert_local_action_to_global(action)
        # Calculate joint velocities from end effector velocities/or if ik is false, just use the action
        self.ur5.action = self.calculate_joint_velocities_from_ee_constrained(self.action)
        singularity = self.ur5.set_joint_velocities(self.ur5.action)

        # Step simulation
        for i in range(self.num_control_simulation_steps):
            self.pyb.con.stepSimulation()

            # print(self.pyb_con.con.getJointStateMultiDof(self.ur5.ur5_robot, self.ur5.end_effector_index))
            # if self.renders: time.sleep(1./240.)

        # Need observations before reward
        self.set_extended_observation()

        # Compute reward
        current_pose = np.hstack((self.observation_info['achieved_pos'], self.observation_info['achieved_or_quat']))
        previous_pose = np.hstack((self.prev_observation_info['achieved_pos'],
                                   self.prev_observation_info['achieved_or_quat']))

        current_pose_eebase = np.hstack((self.observation_info['achieved_eebase_pos'],
                                         self.observation_info['achieved_eebase_or_quat']))
        previous_pose_eebase = np.hstack((self.prev_observation_info['achieved_eebase_pos'],
                                          self.prev_observation_info['achieved_eebase_or_quat']))
        reward, reward_infos = self.compute_reward(self.observation_info['desired_pos'], current_pose,
                                                   previous_pose, current_pose_eebase, previous_pose_eebase,
                                                   singularity, None)

        self.sum_reward += reward

        # self.debug_line = self.pyb_con.con.addUserDebugLine(self.achieved_pos, self.desired_pos, [0, 0, 1], 20)
        self.step_counter += 1
        self.global_step_counter += 1

        # Check if task is done
        done, terminate_info = self.is_task_done()  # done is for gym loggin -> custom_callback

        # Truncated is when the episode is terminated due to time limit,
        # truncated is used to add estimate of reward at the end of the episode to boost training
        # check sb3 docs for more info
        truncated = terminate_info['time_limit_exceeded']
        terminated = terminate_info['goal_achieved_terminate']

        infos = self.get_infos(terminated, truncated)

        infos.update(reward_infos)
        # return self.observation, reward, done, infos
        # v26
        return self.observation, reward, terminated, truncated, infos

    def render(self, mode=None) -> NDArray:  # type: ignore
        sphere = -1
        if "record" in self.name:
            # add sphere
            sphere = self.pyb.add_sphere(radius=0.02, pos=self.observation_info["desired_pos"], rgba=[1, 0, 0, 1], )

        img_rgbd = self.pyb.get_image_at_curr_pose(type='viz')
        img_rgb, _ = self.pyb.seperate_rgbd_rgb_d(img_rgbd, self.cam_height, self.cam_width)
        if mode == "human":
            import cv2
            cv2.imshow("img", (img_rgb * 255).astype(np.uint8))
            cv2.waitKey(1)
        if "record" in self.name:
            # remove sphere
            self.pyb.con.removeBody(sphere)

        return img_rgb

    def close(self) -> None:
        self.pyb.con.disconnect()

    def compute_deprojected_point_mask(self):
        # TODO: Make this function nicer
        # Function. Be Nice.

        point_mask = np.zeros((self.pyb.cam_height, self.pyb.cam_width), dtype=np.float32)

        proj_matrix = np.asarray(self.pyb.proj_mat).reshape([4, 4], order="F")
        view_matrix = np.asarray(
            self.ur5.get_view_mat_at_curr_pose(pan=self.cam_pan, tilt=self.cam_tilt, xyz_offset=self.cam_xyz_offset
                                               )).reshape([4, 4], order="F")
        projection = proj_matrix @ view_matrix @ np.array(
            [self.tree_goal_pos[0], self.tree_goal_pos[1], self.tree_goal_pos[2], 1])
        # Normalize by w
        projection = projection / projection[3]

        # if projection within 1,-1, set point mask to 1
        if projection[0] < 1 and projection[0] > -1 and projection[1] < 1 and projection[1] > -1:
            projection = (projection + 1) / 2
            row = self.pyb.cam_height - 1 - int(projection[1] * (self.pyb.cam_height))
            col = int(projection[0] * self.pyb.cam_width)
            radius = 5  # TODO: Make this a variable proportional to distance
            # modern scikit uses a tuple for center
            rr, cc = disk((row, col), radius)
            point_mask[np.clip(0, rr, self.pyb.cam_height - 1), np.clip(0, cc,
                                                                        self.pyb.cam_width - 1)] = 1  # TODO: This is a hack, numbers shouldnt exceed max and min anyways

        # resize point mask to algo_height, algo_width
        point_mask_resize = cv2.resize(point_mask, dsize=(self.algo_width, self.algo_height))
        point_mask = np.expand_dims(point_mask_resize, axis=0).astype(np.float32)
        return point_mask

    def set_extended_observation(self) -> dict:
        """
        The observations are the current position, the goal position, the current orientation, the current depth
        image, the current joint angles and the current joint velocities
        """
        # TODO: define all these dict as named tuples/dict
        self.prev_observation_info = copy.deepcopy(self.observation_info)
        if 'achieved_pos' not in self.prev_observation_info.keys():
            self.prev_observation_info['achieved_pos'] = self.ur5.init_pos_ee[0]
            self.prev_observation_info['achieved_or_quat'] = self.ur5.init_pos_ee[1]
            self.prev_observation_info['achieved_eebase_pos'] = self.ur5.init_pos_eebase[0]
            self.prev_observation_info['achieved_eebase_or_quat'] = self.ur5.init_pos_eebase[1]

        tool_pos, tool_orient = self.ur5.get_current_pose(self.ur5.end_effector_index)
        tool_base_pos, tool_base_orient = self.ur5.get_current_pose(self.ur5.success_link_index)

        achieved_vel, achieved_ang_vel = self.ur5.get_current_vel(self.ur5.end_effector_index)

        achieved_pos = np.array(tool_pos).astype(np.float32)
        achieved_or_quat = np.array(tool_orient).astype(np.float32)

        achieved_tool_base_pos = np.array(tool_base_pos).astype(np.float32)
        achieved_tool_base_orient = np.array(tool_base_orient).astype(np.float32)

        desired_pos = np.array(self.tree_goal_pos).astype(np.float32)

        joint_angles = np.array(self.ur5.get_joint_angles()).astype(np.float32)

        init_pos_ee = np.array(self.ur5.init_pos_ee[0]).astype(np.float32)
        init_or_ee = np.array(self.ur5.init_pos_ee[1]).astype(np.float32)

        rgb, _ = self.pyb.get_rgbd_at_cur_pose('robot',
                                               self.ur5.get_view_mat_at_curr_pose(pan=self.cam_pan, tilt=self.cam_tilt,
                                                                                  xyz_offset=self.cam_xyz_offset))
        if 'rgb' not in self.observation:
            prev_rgb = np.zeros((self.pyb.cam_height, self.pyb.cam_width, 3))
        else:
            prev_rgb = self.observation['rgb']

        point_mask = self.compute_deprojected_point_mask()

        encoded_joint_angles = np.hstack((np.sin(joint_angles), np.cos(joint_angles)))

        pointing_cosine_sim = self.reward.compute_pointing_cos_sim(achieved_pos, desired_pos, achieved_or_quat,
                                                                   self.tree_goal_or)
        perpendicular_cosine_sim = self.reward.compute_perpendicular_cos_sim(achieved_or_quat, self.tree_goal_or)

        # Just infos to be used in the reward function/other methods
        self.observation_info['desired_pos'] = desired_pos
        self.observation_info['achieved_pos'] = achieved_pos
        self.observation_info['achieved_or_quat'] = achieved_or_quat
        # self.observation_info['rgb'] = rgb
        self.observation_info['pointing_cosine_sim'] = abs(pointing_cosine_sim)
        self.observation_info['perpendicular_cosine_sim'] = abs(perpendicular_cosine_sim)
        self.observation_info['target_distance'] = np.linalg.norm(achieved_pos - desired_pos)
        self.observation_info['achieved_eebase_pos'] = achieved_tool_base_pos
        self.observation_info['achieved_eebase_or_quat'] = achieved_tool_base_orient

        # Actual observation - All wrt base
        ############TODO: Check if this is correct
        t_bw, r_bw = self.pyb.con.invertTransform(self.ur5.init_pos_base[0], self.ur5.init_pos_base[1])
        achieved_pos_b, achieved_or_quat_b = self.pyb.con.multiplyTransforms(t_bw, r_bw, achieved_pos, achieved_or_quat)
        init_pos_ee_b, init_or_ee_b = self.pyb.con.multiplyTransforms(t_bw, r_bw, init_pos_ee, init_or_ee)
        desired_pos_b, desired_or_b = self.pyb.con.multiplyTransforms(t_bw, r_bw, desired_pos, [0, 0, 0, 1])
        achieved_or_mat_b = np.array(self.pyb.con.getMatrixFromQuaternion(achieved_or_quat_b)).reshape(3, 3)
        achieved_or_b_6d = achieved_or_mat_b[:, :2].reshape(6, ).astype(np.float32)
        self.observation['achieved_goal'] = np.array(achieved_pos_b) - np.array(init_pos_ee_b)
        self.observation['desired_goal'] = np.array(desired_pos_b) - np.array(init_pos_ee_b)
        self.observation['relative_distance'] = np.array(achieved_pos_b) - np.array(desired_pos_b)
        # Convert orientation into 6D form for continuity
        self.observation['achieved_or'] = achieved_or_b_6d
        # Image stuff
        #################################################
        self.observation['rgb'] = rgb
        self.observation['prev_rgb'] = prev_rgb
        self.observation['point_mask'] = point_mask
        # Convert joint angles to sin and cos
        self.observation['joint_angles'] = encoded_joint_angles
        # self.observation[
        #     'joint_velocities'] = self.ur5.action  # Check the name of this variable and figure where it is set
        # Action actually achieved

        self.observation['prev_action_achieved'] = np.hstack((achieved_vel, achieved_ang_vel))

        # Privileged critic
        # Add cosine sim perp and point
        self.observation['critic_pointing_cosine_sim'] = np.array(pointing_cosine_sim).astype(np.float32).reshape(1, )
        self.observation['critic_perpendicular_cosine_sim'] = np.array(perpendicular_cosine_sim).astype(
            np.float32).reshape(1, )

        if "record" in self.name:
            # add sphere
            sphere = self.pyb.add_sphere(radius=0.005, pos=self.observation_info["desired_pos"], rgba=[1, 0, 0, 1], )
            rgb, _ = self.pyb.get_rgbd_at_cur_pose('robot',
                                                   self.ur5.get_view_mat_at_curr_pose(pan=self.cam_pan,
                                                                                      tilt=self.cam_tilt,
                                                                                      xyz_offset=self.cam_xyz_offset))
            # remove sphere
            self.observation_info['rgb'] = rgb
            self.pyb.con.removeBody(sphere)

        return self.observation

    def is_task_done(self) -> Tuple[bool, dict]:
        # Terminate if time limit exceeded
        # NOTE: need to call compute_reward before this to check termination!
        time_limit_exceeded = self.step_counter >= self.maxSteps
        if "test" in self.name or self.eval:
            goal_achieved_terminate = self.is_goal_state
            done = time_limit_exceeded or goal_achieved_terminate
        else:
            goal_achieved_terminate = self.is_goal_state  # Do not terminate during training
            done = time_limit_exceeded or goal_achieved_terminate  # (self.terminated is True or self.step_counter > self.maxSteps)
        terminate_info = {"time_limit_exceeded": time_limit_exceeded,
                          "goal_achieved_terminate": goal_achieved_terminate}
        if self.verbose > 1:
            print("DEBUG: Time limit exceeded: ", time_limit_exceeded)
            print("DEBUG: Goal achieved: ", goal_achieved_terminate)
        return done, terminate_info

    def compute_reward(self, desired_goal, achieved_pose,
                       previous_pose, achieved_pose_eebase, previous_pose_eebase, singularity: bool,
                       info: Optional[dict]) -> Tuple[float, dict]:

        achieved_pos = achieved_pose[:3]
        achieved_or = achieved_pose[3:]
        desired_pos = desired_goal
        previous_pos = previous_pose[:3]
        previous_or = previous_pose[3:]
        achieved_pos_eebase = achieved_pose_eebase[:3]
        previous_pos_eebase = previous_pose_eebase[:3]
        achieved_or_eebase = achieved_pose_eebase[3:]
        previous_or_eebase = previous_pose_eebase[3:]
        reward = 0.0

        self.collisions_acceptable = 0
        self.collisions_unacceptable = 0
        if self.verbose > 1:
            _ = self.pyb.add_debug_item('line', 'step', lineFromXYZ=achieved_pos, lineToXYZ=desired_pos,
                                        lineColorRGB=[0, 0, 1], lineWidth=20)
            self.pyb.add_debug_item('sphere', 'step', lineFromXYZ=self.tree_goal_pos,
                                    lineToXYZ=[self.tree_goal_pos[0] + 0.005, self.tree_goal_pos[1] + 0.005,
                                               self.tree_goal_pos[2] + 0.005],
                                    lineColorRGB=[1, 0, 0],
                                    lineWidth=400)
            self.pyb.add_debug_item('line', 'step', lineFromXYZ=self.tree_goal_pos - 50 * self.tree_goal_or,
                                    lineToXYZ=self.tree_goal_pos + 50 * self.tree_goal_or, lineColorRGB=[1, 0, 0],
                                    lineWidth=400)
        # _ = self.pyb_con.add_debug_item('line', 'step', lineFromXYZ=previous_pos, lineToXYZ=desired_pos,
        #                             lineColorRGB=[0, 0, 1], lineWidth=20)

        # Calculate rewards
        reward += self.reward.calculate_distance_reward(achieved_pos, desired_pos)
        reward += self.reward.calculate_movement_reward(achieved_pos, previous_pos, desired_pos)
        point_reward, point_cosine_sim = self.reward.calculate_pointing_orientation_reward(achieved_pos_eebase,
                                                                                           desired_pos,
                                                                                           achieved_or_eebase,
                                                                                           previous_pos_eebase,
                                                                                           previous_or_eebase,
                                                                                           self.tree_goal_or)
        reward += point_reward

        perp_reward, perp_cosine_sim = self.reward.calculate_perpendicular_orientation_reward(achieved_or_eebase,
                                                                                              previous_or_eebase,
                                                                                              self.tree_goal_or)
        reward += perp_reward
        condition_number = self.ur5.get_condition_number()
        reward += self.reward.calculate_condition_number_reward(condition_number)

        # If is successful
        self.is_goal_state = self.is_state_successful(achieved_pos, desired_pos, perp_cosine_sim, point_cosine_sim)
        reward += self.reward.calculate_termination_reward(self.is_goal_state)
        is_collision, collision_info = self.ur5.check_collisions(self.collision_object_ids)
        reward += self.reward.calculate_acceptable_collision_reward(collision_info)
        reward += self.reward.calculate_unacceptable_collision_reward(collision_info)

        # check collisions:
        if is_collision:
            self.log_collision_info(collision_info)

        reward += self.reward.calculate_slack_reward()
        reward += self.reward.calculate_velocity_minimization_reward(self.action)

        return reward, self.reward.reward_info

    def is_state_successful(self, achieved_pos, desired_pos, orientation_perp_value, orientation_point_value):
        terminated = False
        # TODO: Success only if the collision is in the branch plane
        is_success_collision_spur = self.ur5.check_success_collision(self.collision_object_ids['SPUR'])
        is_success_collision_water_branch = self.ur5.check_success_collision(self.collision_object_ids['WATER_BRANCH'])
        is_success_collision = is_success_collision_spur or is_success_collision_water_branch
        dist_from_target = np.linalg.norm(achieved_pos - desired_pos)
        if is_success_collision and dist_from_target < self.distance_threshold:
            if (orientation_perp_value > self.angle_threshold_perp_cosine) and (
                    orientation_point_value > self.angle_threshold_point_cosine):
                terminated = True
                if self.verbose > 1:
                    print("DEBUG: Task successful")
        if self.verbose > 1:
            print("DEBUG: Distance from target: ", dist_from_target)
            print("DEBUG: Perpendicular cosine sim: ", orientation_perp_value)
            print("DEBUG: Pointing cosine sim: ", orientation_point_value)
            print("DEBUG: Acceptable collision: ", is_success_collision)
        return terminated

    def log_collision_info(self, collision_info):
        if collision_info['collisions_acceptable']:
            self.collisions_acceptable += 1
        elif collision_info['collisions_unacceptable']:
            self.collisions_unacceptable += 1

    # TODO: Create new class for pruning env


class PruningEnvRRT(PruningEnv):

    def __init__(self, tree_urdf_path: str, tree_obj_path: str, tree_labelled_path: str, renders: bool = False,
                 max_steps: int = 500,
                 distance_threshold: float = 0.05, angle_threshold_perp: float = 0.52,
                 angle_threshold_point: float = 0.52,
                 tree_count: int = 9999, cam_width: int = 424, cam_height: int = 240,
                 algo_width: int = 224, algo_height: int = 224,
                 evaluate: bool = False, num_points: Optional[int] = None, action_dim: int = 12,
                 name: str = "PruningEnv", action_scale: int = 1, movement_reward_scale: int = 1,
                 distance_reward_scale: int = 1, condition_reward_scale: int = 1, terminate_reward_scale: int = 1,
                 collision_reward_scale: int = 1, slack_reward_scale: int = 1,
                 perpendicular_orientation_reward_scale: int = 1, pointing_orientation_reward_scale: int = 1,
                 curriculum_distances: Tuple = (0.8,), curriculum_level_steps: Tuple = (),
                 use_ik: bool = True, make_trees: bool = False,
                 ur5_pos=[0, 0, 0], ur5_or=[0, 0, 0, 1], randomize_ur5_pose: bool = False,
                 randomize_tree_pose: bool = False,
                 verbose=1
                 ) -> None:

            super().__init__(tree_urdf_path, tree_obj_path, tree_labelled_path, renders, max_steps, distance_threshold,
                            angle_threshold_perp, angle_threshold_point, tree_count, cam_width, cam_height, algo_width,
                            algo_height, evaluate, num_points, action_dim, name, action_scale, movement_reward_scale,
                            distance_reward_scale, condition_reward_scale, terminate_reward_scale, collision_reward_scale,
                            slack_reward_scale, perpendicular_orientation_reward_scale, pointing_orientation_reward_scale,
                            curriculum_distances, curriculum_level_steps, use_ik, make_trees, ur5_pos, ur5_or,
                            randomize_ur5_pose, randomize_tree_pose, verbose)

    def generate_goal_pos(self):
        # self.pyb.remove_debug_items("step")
        branch_normal = self.tree_goal_normal / np.linalg.norm(self.tree_goal_normal)
        branch_parallel = self.tree_goal_or / np.linalg.norm(self.tree_goal_or)
        forward = np.cross(branch_parallel, branch_normal)
        # Get a vector in plane of forward and right using linear combination
        rotation_matrix = np.column_stack((branch_parallel, branch_normal, forward))
        rotation_matrix = R.from_matrix(rotation_matrix).as_matrix()

        rotation_axis_x = rotation_matrix[:, 0]
        #How much to rotate about this axis to make same as branch normal?
        #self.ur5.init_pos_ee[1] as rotation matrix
        rot_ee = self.pyb.con.getMatrixFromQuaternion(self.ur5.init_pos_ee[1])
        rot_ee = np.array(rot_ee).reshape(3, 3)
        rot_pointing = rot_ee[:, 2]

        #Project rot_pointing perpendicular to rotation_axis_x
        rot_pointing = rot_pointing - np.dot(rot_pointing, rotation_axis_x) * rotation_axis_x
        rot_pointing = rot_pointing / np.linalg.norm(rot_pointing)
        #Minimize the angle between the branch normal and the rotation matrix [2] axis
        cos_theta = np.dot(rot_pointing, rotation_matrix[:, 2])
        sin_theta = np.linalg.norm(np.cross(rot_pointing, rotation_matrix[:, 2]))
        theta = np.arctan2(sin_theta, cos_theta)
        # if np.dot(cross_product, rotation_axis_x) < 0:
        #     theta = theta + np.pi
        # print(np.rad2deg(theta), theta)
        # if theta > np.pi:
        #     theta = theta - np.pi
        # print(np.rad2deg(theta), theta)
        #Make this concentration lower to make the solutions more random as num_attempts increases.
        concentration = 0.85

        rotation_angle_x = vonmises(loc = theta, kappa = concentration).rvs(1)
        # rotation_angle_x = np.random.uniform(0, 2 * np.pi)
        random_rotation_x = R.from_rotvec(rotation_angle_x * rotation_axis_x).as_matrix()

        rotation_axis_y = rotation_matrix[:, 1]
        rotation_angle_y = np.random.uniform(0, self.angle_threshold_perp)
        random_rotation_y = R.from_rotvec(rotation_angle_y * rotation_axis_y).as_matrix()
        # random_rotation_y = np.eye(3)
        rotation_axis_z = rotation_matrix[:, 2]
        rotation_angle_z = np.random.uniform(0, self.angle_threshold_point)
        random_rotation_z = R.from_rotvec(rotation_angle_z * rotation_axis_z).as_matrix()
        # random_rotation_z = np.eye(3)

        random_rotation = random_rotation_z @ random_rotation_y @ random_rotation_x
        rotation_matrix = random_rotation @ rotation_matrix
        # print(np.dot(rotation_matrix[:, 2], rot_pointing))
        # self.pyb.visualize_rot_mat(rotation_matrix, self.tree_goal_pos)
        # self.pyb.visualize_rot_mat(rot_ee, self.tree_goal_pos)
        # self.pyb.visualize_rot_mat(rot_ee, self.ur5.init_pos_ee[0])
        # input()

        # time.sleep(1)
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()
        return quaternion, forward, rotation_matrix

    def sample_goal(self, config=False, offset=0.03):
        orientation, forward, rot = self.generate_goal_pos()
        init_vec = np.array([0, 0, 1])
        init_vec = rot @ init_vec
        position = self.tree_goal_pos - offset * init_vec
        if config:
            joint_angles = self.ur5.calculate_ik(position, orientation)
            return joint_angles
        return position, orientation

    def get_different_ik_results(self, total_attempts, timing=None, check_collision=True):
        """This is a generator that yields different ik results"""
        attempts = 0
        while attempts < total_attempts:
            start_time = time.time()
            self.pyb.remove_debug_items("step")
            self.ur5.reset_ur5_arm()
            self.pyb.con.stepSimulation()
            collision = False

            goal, goal_orientation = self.sample_goal()

            self.ur5.set_collision_filter_tree(self.collision_object_ids)
            offset = np.random.uniform(-0.1, 0.1, 3)
            # scale the offset to be 15cm away from the goal
            offset = 0.15 * offset / np.linalg.norm(offset)
            initial_guess = self.ur5.calculate_ik(goal + offset, goal_orientation)
            # if no_collision:
            self.ur5.set_joint_angles_no_collision(initial_guess)
            self.pyb.con.stepSimulation()
            # else:
            #     self.ur5.set_joint_angles(initial_guess)
            #     for j in range(30):
            #         self.pyb.con.stepSimulation()

            joint_angles = self.ur5.calculate_ik(goal, goal_orientation)
            self.ur5.set_joint_angles_no_collision(joint_angles)
            # else:
            #     self.ur5.set_joint_angles(joint_angles)
            #     for j in range(30):
            #         self.pyb.con.stepSimulation()
            # self.activate_tree(self.pyb)
            # self.pyb.con.stepSimulation()
            # print("Actual", self.ur5.get_joint_angles())

            # if not np.isclose(joint_angles, self.ur5.get_joint_angles(), atol=1e-1).all():
            #     # print("Cant set joint angles")
            #     # print(joint_angles, self.ur5.get_joint_angles())
            #     attempts += 1
            #     continue
            if check_collision:
                self.ur5.unset_collision_filter_tree(self.collision_object_ids)
                self.pyb.con.stepSimulation()
                collision_info = self.ur5.check_collisions(self.collision_object_ids)

                if collision_info[1]['collisions_unacceptable']:
                    collision = True
                    if self.verbose > 1:
                        print("DEBUG: Collision")
            else:
                self.pyb.con.stepSimulation()
                collision = False
            if not collision:
                current_ee_pose = self.ur5.get_current_pose(self.ur5.end_effector_index)
                delta_q = self.pyb.con.getDifferenceQuaternion(current_ee_pose[1], goal_orientation)
                delta_angle = 2 * np.arccos(delta_q[3])
                if np.linalg.norm(current_ee_pose[0] - goal) < 0.05 and delta_angle < 0.1:
                    if self.verbose > 1:
                        print("DEBUG: Found solution")
                    yield joint_angles, goal, goal_orientation
                    # solutions.append((joint_angles, goal, orientation))
                else:
                    if self.verbose > 1:
                        print("DEBUG: Not close enough")

            self.ur5.reset_ur5_arm()
            attempts += 1
            runtime = time.time() - start_time
            timing['time_find_end_config'] += runtime
            timing['attempts'] = attempts

        if attempts == total_attempts:
            if self.verbose > 1:
                print("DEBUG: No valid solutions found")
            yield None
            return
            # return 0
        # return solutions

    def render_save_image(self, goal):
        sphere = self.pyb.add_sphere(radius=0.01, pos=goal, rgba=[1, 0, 0, 1], )
        render = np.array(self.render()) * 255
        render = render.astype(np.uint8)
        render = cv2.resize(render, (512, 512), interpolation=cv2.INTER_NEAREST)
        robot_img, _ = self.pyb.get_rgbd_at_cur_pose('robot',
                                                     self.ur5.get_view_mat_at_curr_pose(pan=self.cam_pan,
                                                                                        tilt=self.cam_tilt,
                                                                                        xyz_offset=self.cam_xyz_offset))
        robot_img = robot_img * 255
        robot_img = robot_img.astype(np.uint8)
        robot_img = cv2.resize(robot_img, (512, 512), interpolation=cv2.INTER_NEAREST)
        render = np.hstack((render, robot_img))
        self.pyb.con.removeBody(sphere)
        return render

    def set_dataset(self, dataset):
        self.dataset = dataset


    def run_naive_ik(self, planner, file_path, save_video=False, shortcutting=False): #Consistent with the other run functions
        for i in range(len(self.dataset)):
            self.reset()

            timing = {'time_find_end_config': 0, 'time_find_path': 0, 'time_total': 0, 'attempts': 0}
            solutions = self.get_different_ik_results(total_attempts=60, timing=timing)
            success = True
            total_attempts = 0
            total_solutions = 0
            for sol in solutions:
                if sol is not None:
                    success = True
                    current_joint_angles = np.array(self.ur5.get_joint_angles())
                    goal_joint_angles, goal_pos, goal_or = sol
                    linear_diff = (np.array(goal_joint_angles) - np.array(current_joint_angles))/500

                    for j in range(501):
                        self.ur5.set_joint_angles(current_joint_angles + j * linear_diff)
                        self.pyb.con.stepSimulation()
                        is_collision, collision_info = self.ur5.check_collisions(self.collision_object_ids)
                        if is_collision:
                            success = False
                            break
                    if success:
                        j_a = np.array(self.ur5.get_joint_angles())
                        if np.isclose(j_a, np.array(goal_joint_angles), atol=1e-2).all():
                            break
                        else:
                            success = False


            if sol is None:
                success = False
            result = {"success": success}#, "total_solutions": total_solutions}
            result = pd.DataFrame([result])
            self.append_row_to_csv(result, file_path)

        # print(total_solutions/total_attempts)
    def run_baseline(self, planner, file_path, save_video=False, shortcutting=False):
        for i in range(len(self.dataset)):
            print("Starting", i, len(self.dataset))
            self.reset()
            # callback.update_tree_properties(self.id)
            tree_info = self.dataset[i]
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal = tree_info
            self.set_tree_properties(tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                     point_branch_normal=current_branch_normal)
            if planner == "rrt_connect":
                path, tree_info, goal_orientation, timing = self.run_rrt_connect(save_video=save_video, shortcutting=shortcutting)
            elif planner == "informed_rrt_star":
                path, tree_info, goal_orientation, timing = self.run_rrt_star(save_video=save_video, informed = True, shortcutting = shortcutting)
            elif planner == "rrt_star":
                path, tree_info, goal_orientation, timing = self.run_rrt_star(save_video=save_video, informed = False, shortcutting = shortcutting)
            elif planner == "straight_line_ik":
                path, tree_info, goal_orientation, timing = self.run_straight_line_ik(save_video=save_video)
            elif planner == "reachability":
                path, tree_info, goal_orientation, timing = self.run_reachability(save_video=save_video)
            else:
                raise ValueError("Planner not found")

            goal_pos = tree_info[1]
            goal_or = tree_info[2]

            success = isinstance(path, list)
            if success:
                fail_mode = 1
                #path = self.append_goal_config(path)
            else:
                fail_mode = path
            print("Completed", i, success, fail_mode)
            result = {"pointx": goal_pos[0], "pointy": goal_pos[1], "pointz": goal_pos[2], "or_x": goal_or[0],
                      "or_y": goal_or[1], "or_z": goal_or[2], "is_success": success, "fail_mode": fail_mode}#, "path": path, "tree_info": tree_info}
            result.update(timing)

            # write to existing file
            result = pd.DataFrame([result])
            self.append_row_to_csv(result, file_path)
            # result_df = pd.concat([result_df, result], ignore_index=True)
        self.pyb.con.disconnect()
        return

    def append_row_to_csv(self, row, file_path):
        # Create a file lock for thread/process safety
        lock_path = file_path + '.lock'
        with FileLock(lock_path):
            # Check if file exists
            if not os.path.exists(file_path):
                # Write the row and header to a new file
                row.to_csv(file_path, index=False, mode='w', header=True)
            else:
                # Append the row without header
                row.to_csv(file_path, index=False, mode='a', header=False)

    def run_reachability(self, save_video=False):
        #Return true if my arm can reach the given pose without considering collisions
        timing = {'time_find_end_config': 0, 'time_find_path': 0, 'time_total': 0}
        solutions = self.get_different_ik_results(total_attempts=100, timing=timing)
        tree_info = [self.tree_urdf, self.tree_goal_pos, self.tree_goal_or, self.tree_orientation, self.tree_scale,
                        self.tree_pos, self.tree_goal_normal]
        for i in solutions:
            if i is None:
                return ResultMode.NO_PATH, tree_info, None, timing
            # self.ur5.set_joint_angles(i)
            # for j in range(50):
            #     self.pyb.con.stepSimulation()
            # pos, goal_or = self.ur5.get_current_pose(self.ur5.end_effector_index)
            # if np.linalg.norm(np.array(pos) - np.array(self.tree_goal_pos)) < 0.05:
            else:
                goal, goal_pos, goal_or = i
                self.ur5.set_joint_angles_no_collision(goal)
                for j in range(2):
                    self.pyb.con.stepSimulation()

                condition_number = self.ur5.get_condition_number()
                return condition_number, tree_info, None, timing



    def run_rrt_star(self, save_video=False, informed = True, shortcutting = False):
        planner = "informed_rrt_star" if informed else "rrt_star"
        timing = {'time_find_end_config': 0, 'time_find_path': 0, 'time_total': 0}
        tree_info = [self.tree_urdf, self.tree_goal_pos, self.tree_goal_or, self.tree_orientation, self.tree_scale,
                     self.tree_pos, self.tree_goal_normal]
        controllable_joints = [3, 4, 5, 6, 7, 8]
        distance_fn = get_distance_fn(self.ur5.ur5_robot, controllable_joints)
        sample_position = get_sample_fn(self.ur5.ur5_robot, controllable_joints)
        extend_fn = get_extend_fn(self.ur5.ur5_robot, controllable_joints)
        goal_fn = self.sample_goal
        is_goal_fn = self.is_state_successful_rrt
        collision_objects = []
        for key, val in self.collision_object_ids.items():
            # if key != "SPUR":
            collision_objects.append(val)
        collision_fn = get_collision_fn(self.ur5.ur5_robot, controllable_joints, collision_objects)
        start_find_path = time.time()
        if informed:
            path = informed_rrt_star_multi_goal(self.ur5.get_joint_angles(), goal_fn, distance_fn, sample_position,
                                            extend_fn, collision_fn, radius=0.4, is_goal_fn = is_goal_fn, max_iterations=5000)
        else:
            path = rrt_star_multi_goal(self.ur5.get_joint_angles(), goal_fn, distance_fn, sample_position,
                            extend_fn, collision_fn, radius=0.4, is_goal_fn = is_goal_fn, max_iterations=5000, informed = False)
        if path is not None:
            if save_video:
                self.baseline_save_video(path, planner, tree_info[1])
        timing['time_find_path'] = time.time() - start_find_path - timing['time_find_end_config']
        timing['time_total'] = timing['time_find_end_config'] + timing['time_find_path']
        if path is None:
            print("No valid path found")
            return ResultMode.NO_PATH, tree_info, None, timing

        if shortcutting:
            path = smooth_path(path, extend_fn, collision_fn, distance_fn, sample_fn=sample_position, coarse_waypoints = False)
        goal_config = path[-1]
        self.ur5.set_joint_angles(goal_config)
        for j in range(50):
            self.pyb.con.stepSimulation()
        pos, goal_or = self.ur5.get_current_pose(self.ur5.end_effector_index)
        return (path, tree_info, goal_or, timing)


    def shortcut_and_refine_path(self, path, extend_fn, collision_fn, distance_fn, sample_fn, controllable_joints, task_threshold = 0.05):
        path = smooth_path(path, extend_fn, collision_fn, distance_fn, sample_fn=sample_fn, coarse_waypoints = True)
        path = self.refine_waypoints_in_task_space(self.ur5.ur5_robot, controllable_joints, path, task_threshold=task_threshold)
        return path

    def baseline_save_video(self, path, planner, goal):
        frames = []
        self.pyb.remove_debug_items("step")
        self.pyb.remove_debug_items("reset")
        self.reset()
        frames.append(self.render_save_image(goal))
        for j_a in path:
            self.ur5.set_joint_angles_no_collision(j_a)
            frames.append(self.render_save_image(goal))
        if self.verbose > 0:
            print("INFO: Saving video " + "results_rrt_gifs/{}_{}.gif".format(planner, time.time()))
        imageio.mimsave("results_rrt_gifs/{}_{}.gif".format(planner, time.time()),
                        frames, loop=0)

    def get_planning_fns(self):
        controllable_joints = [3, 4, 5, 6, 7, 8]
        distance_fn = get_distance_fn(self.ur5.ur5_robot, controllable_joints)
        sample_position = get_sample_fn(self.ur5.ur5_robot, controllable_joints)
        extend_fn = get_extend_fn(self.ur5.ur5_robot, controllable_joints)
        collision_objects = []
        for key, val in self.collision_object_ids.items():
            # if key != "SPUR":
            collision_objects.append(val)
        collision_fn = get_collision_fn(self.ur5.ur5_robot, controllable_joints, collision_objects)
        return distance_fn, sample_position, extend_fn, collision_fn

    def run_straight_line_ik(self, save_video=False):
        timing = {'time_find_end_config': 0, 'time_find_path': 0, 'time_total': 0}
        solutions = self.get_different_ik_results(total_attempts=60, timing=timing)
        tree_info = [self.tree_urdf, self.tree_goal_pos, self.tree_goal_or, self.tree_orientation, self.tree_scale,
                     self.tree_pos, self.tree_goal_normal]
        path = None
        solution_found = False
        for i in solutions:
            if i is None:
                break
            path = []
            solution_found = True
            self.ur5.reset_ur5_arm()
            self.pyb.con.stepSimulation()
            start = self.ur5.get_joint_angles()
            goal, goal_pos, goal_or = i
            linear_diff = (np.array(goal) - np.array(start))/50
            for j in range(51):
                self.ur5.set_joint_angles(np.array(start) + j * linear_diff)
                for _ in range(20):
                    self.pyb.con.stepSimulation()
                is_collision, collision_info = self.ur5.check_collisions(self.collision_object_ids)
                if is_collision:
                    path = None
                    break
                else:
                    path.append(self.ur5.get_joint_angles())
            if path is None:
                continue
            else:
                if not np.isclose(path[-1], goal, atol=5e-2).all():
                    path = None
                    continue

                elif save_video:
                    self.baseline_save_video(path, "straight_line", goal_pos)
                break

        if not solution_found:
            print("No valid solutions found")
            return ResultMode.NO_SOLUTION, tree_info, None, timing

        if path is None or len(path) == 0:
            print("No valid path found")
            return ResultMode.NO_PATH, tree_info, None, timing

        print("Path found")
        return (path, tree_info, goal_or, timing)
    def run_rrt_connect(self, save_video=False, shortcutting=False):
        timing = {'time_find_end_config': 0, 'time_find_path': 0, 'time_total': 0}
        solutions = self.get_different_ik_results(total_attempts=100, timing=timing)

        tree_info = [self.tree_urdf, self.tree_goal_pos, self.tree_goal_or, self.tree_orientation, self.tree_scale,
                     self.tree_pos, self.tree_goal_normal]


        distance_fn, sample_position, extend_fn, collision_fn = self.get_planning_fns()
        controllable_joints = [3, 4, 5, 6, 7, 8]

        start_find_path = time.time()
        path = None
        solution_found = False
        for i in solutions:
            if i is None:
                break
            solution_found = True
            self.ur5.reset_ur5_arm()
            self.pyb.con.stepSimulation()
            start = self.ur5.get_joint_angles()
            goal, goal_pos, goal_or = i
            path = pp.rrt_connect(start, goal, distance_fn, sample_position, extend_fn, collision_fn,
                                  max_iterations=10000)

            if path is None:
                continue
            else:
                if save_video:
                    self.baseline_save_video(path, "rrt_connect", tree_info[1])
                break

        timing['time_find_path'] = time.time() - start_find_path - timing['time_find_end_config']
        timing['time_total'] = timing['time_find_end_config'] + timing['time_find_path']

        if not solution_found:
            print("No valid solutions found")
            return ResultMode.NO_SOLUTION, tree_info, None, timing

        if path is None:
            print("No valid path found")
            return ResultMode.NO_PATH, tree_info, None, timing


        if path is not None and shortcutting:
            # path = smooth_path(path, extend_fn, collision_fn, distance_fn, sample_fn=sample_position, coarse_waypoints = True)
            # print("refining waypoints")
            # path = self.refine_waypoints_in_task_space(self.ur5.ur5_robot, controllable_joints, path)
            path = self.shortcut_and_refine_path(path, extend_fn, collision_fn, distance_fn, sample_position, controllable_joints)
        return (path, tree_info, goal_or, timing)


    def task_space_distance(self, positions1, positions2):
        #Forward Kinematics
        self.ur5.set_joint_angles_no_collision(positions1)
        self.pyb.con.stepSimulation()
        pos1, or1 = self.ur5.get_current_pose(self.ur5.end_effector_index)
        self.ur5.set_joint_angles_no_collision(positions2)
        self.pyb.con.stepSimulation()
        pos2, or2 = self.ur5.get_current_pose(self.ur5.end_effector_index)
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    def get_refine_fn_task(self, body, joints, task_threshold):
        #Decrease step size as gettin closer to the goal.
        difference_fn = get_difference_fn(body, joints)
        def fn(q1, q2):
            yield q1
            num_steps = int(np.ceil(np.linalg.norm(np.divide(difference_fn(q2, q1), 0.1), ord=2)))+1
            prev_positions = q1
            for i in range(num_steps):
                positions = (i / (num_steps - 1)) * np.array(difference_fn(q2, q1)) + q1
                if (not (i == (num_steps - 1))) and self.task_space_distance(positions, prev_positions) < task_threshold:
                    continue

                # q = tuple(positions)
                # q = tuple(wrap_positions(body, joints, positions))
                if i == num_steps - 1:
                    yield q2
                else:
                    position_within_threshold = ( (i - 1)/ (num_steps - 1)) * np.array(difference_fn(q2, q1)) + q1
                    yield position_within_threshold
                    prev_positions = position_within_threshold
                # yield q

        return fn

    def refine_waypoints_in_task_space(self, body, joints, waypoints, task_threshold = 0.05):
        refine_fn = self.get_refine_fn_task(body, joints, task_threshold)
        refined_path = []
        for v1, v2 in zip(waypoints, waypoints[1:]):
            refined_path += list(refine_fn(v1, v2))
        return refined_path

    def is_state_successful_rrt(self, config):
        achieved_pos, achieved_or = self.ur5.get_current_pose(self.ur5.end_effector_index)
        desired_pos = self.tree_goal_pos
        orientation_perp_value = self.reward.compute_perpendicular_cos_sim(achieved_or, self.tree_goal_or)
        orientation_point_value = self.reward.compute_pointing_cos_sim(achieved_pos, desired_pos, achieved_or,
                                                                       self.tree_goal_or)
        terminated = False
        dist_from_target = np.linalg.norm(achieved_pos - desired_pos)
        if dist_from_target < self.distance_threshold:
            if (abs(orientation_perp_value) > self.angle_threshold_perp_cosine) and (
                    orientation_point_value > self.angle_threshold_point_cosine):
                terminated = True
                if self.verbose > 1:
                    print("DEBUG: Task successful")
        return terminated

    def append_goal_config(self, path):
        #Set to path[-1]
        self.ur5.set_joint_angles_no_collision(path[-1])
        self.pyb.con.stepSimulation()
        _, orient = self.ur5.get_current_pose(self.ur5.end_effector_index)
        goal_config = self.ur5.calculate_ik(self.tree_goal_pos, orient)
        path.append(goal_config)
        self.ur5.reset_ur5_arm()
        self.pyb.con.stepSimulation()
        return path

    def run_smoothing(self, save_video=False, save_path = None):
        controllable_joints = [3, 4, 5, 6, 7, 8]
        paths_success = self.dataset
        for i in range(len(paths_success)):
            observations = []
            new_observations = []
            rewards = []
            dones = []
            actions = []
            terminated = False
            tree_info = paths_success.iloc[i]['tree_info']
            path = paths_success.iloc[i]['path']

            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal = tree_info

            #Set tree info
            self.pyb.remove_debug_items('step')
            self.pyb.remove_debug_items('reset')
            self.set_tree_properties(tree_urdf=tree_urdf,
                                    point_pos=final_point_pos, point_branch_or=current_branch_or,
                                    tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                    point_branch_normal=current_branch_normal)
            self.pyb.con.stepSimulation()
            if self.verbose > 0:
                print("INFO: Refining path", i)

            distance_fn, sample_position, extend_fn, collision_fn = self.get_planning_fns()

            refined_path = self.shortcut_and_refine_path(path, extend_fn, collision_fn, distance_fn, sample_position,
                                                        controllable_joints, task_threshold=0.05)
            # refined_path = self.append_goal_config(refined_path)
            # refined_path = self.append_goal_config(refined_path)


            ee_vel = self.convert_ja_to_ee_vel(refined_path, control_freq=int(1./self.control_time))
            obs, _ = self.reset()
            obs = copy.deepcopy(obs)
            # Change rgb, prev_rgb, to CHW from HWC. SB3 does this somewhere automatically, find if time
            obs['rgb'] = obs['rgb'].transpose(2, 0, 1)
            obs['prev_rgb'] = obs['prev_rgb'].transpose(2, 0, 1)
            if save_video:
                self.baseline_save_video(refined_path, 'refined_path', final_point_pos)
                self.ur5.reset_ur5_arm()
            count_in_frame = 0
            for i, vel in enumerate(ee_vel):
                if np.isclose(vel, np.zeros(6), atol=0.001).all():
                    continue
                scaled_vel = vel/self.action_scale #Value passed by name, will get unscaled after step
                action = copy.deepcopy(scaled_vel)
                new_obs, reward, terminated, truncated, _ = self.step(scaled_vel)
                #Change rgb, prev_rgb, to CHW from HWC. SB3 does this somewhere automatically, find if time
                new_obs = copy.deepcopy(new_obs)
                new_obs['rgb'] = new_obs['rgb'].transpose(2, 0, 1)
                new_obs['prev_rgb'] = new_obs['prev_rgb'].transpose(2, 0, 1)
                new_observations.append(new_obs)
                observations.append(obs)
                rewards.append(reward)
                dones.append(terminated)
                actions.append(action)

                if (self.observation['point_mask'] > 0).any():
                    count_in_frame += 1
                else:
                    pass
                obs = new_obs

            #Once we are done with the planner trajectory, we run a cartesian planner to get to the cut poitn

            pos, orient = self.ur5.get_current_pose(self.ur5.end_effector_index)
            ee_vel_global = np.zeros(6)
            ee_vel_global[:3] = (np.array(final_point_pos) - np.array(pos))/self.control_time #Velocity needed to reach the goal in 1 sim step
            scaled_vel_global = ee_vel_global #Take at least 3 steps to reach the goal

            ee_vel_local = self.convert_global_action_to_local(scaled_vel_global)
            scale = 1#np.max(np.abs(ee_vel_local))
            if (ee_vel_local > 1).any() or (ee_vel_local < -1).any():
                scale = np.max(np.abs(ee_vel_local))
                ee_vel_local = ee_vel_local/(scale+1e-4)
            #We run the simulation for self.pyb.step_time seconds
            if not terminated:
                for j in range(int(2*int(scale)/self.action_scale)): #At most twice the time needed to reach the goal
                    #Move towards the goal
                    if np.isclose(ee_vel_local, np.zeros(6), atol=0.001).all():
                        dones[-1] = True
                        break
                    action = copy.deepcopy(ee_vel_local)
                    new_obs, reward, terminated, truncated, info = self.step(copy.deepcopy(ee_vel_local))
                    actions.append(action)
                    new_obs = copy.deepcopy(new_obs)
                    # Change rgb, prev_rgb, to CHW from HWC. SB3 does this somewhere automatically using VecEnvWrapper

                    new_obs['rgb'] = new_obs['rgb'].transpose(2, 0, 1)
                    new_obs['prev_rgb'] = new_obs['prev_rgb'].transpose(2, 0, 1)

                    new_observations.append(new_obs)
                    observations.append(obs)
                    rewards.append(reward)
                    if (self.observation['point_mask'] > 0).any():
                        count_in_frame += 1
                    #if last step, done is true


                    if terminated or info['collision_unacceptable_reward'] < 0 or info['collision_acceptable_reward'] < 0:
                        dones.append(True)
                        break
                    if j == int(2*int(scale)/self.action_scale) - 1:
                        dones.append(True)
                    else:
                        dones.append(terminated)

                    obs = new_obs
            print("Length of velo", len(actions))
            save_dict = {"tree_info": tree_info, "observations": observations, "actions": actions, "rewards": rewards, "dones": dones, "trajectory_in_frame": count_in_frame/len(actions), "next_observations": new_observations, "info": info}
            #Calculate optical flow

            with open('expert_trajectories/{}.pkl'.format(time.time()), 'wb') as f:
                pickle.dump(save_dict, f)

            if save_path is not None:
                result = {"path": ee_vel, "len_refined_path": len(refined_path), "trajectory_in_frame": count_in_frame / len(ee_vel),
                            "tree_info": tree_info}
                result = pd.DataFrame([result])
                self.append_row_to_csv(result, save_path)

    def task_space_distance(self, positions1, positions2):
        #Forward Kinematics
        self.ur5.set_joint_angles_no_collision(positions1)
        self.pyb.con.stepSimulation()
        pos1, or1 = self.ur5.get_current_pose(self.ur5.end_effector_index)
        self.ur5.set_joint_angles_no_collision(positions2)
        self.pyb.con.stepSimulation()
        pos2, or2 = self.ur5.get_current_pose(self.ur5.end_effector_index)
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def convert_ja_to_ee_vel(self, path, control_freq = 2):
        ee_vel = []
        for i in range(len(path) - 1):
            joint_velocities = (np.array(path[i+1]) - np.array(path[i]))*control_freq
            self.ur5.set_joint_angles_no_collision(path[i])
            self.pyb.con.stepSimulation()
            jacobian = self.ur5.calculate_jacobian()
            global_ee_vel = np.matmul(jacobian, joint_velocities)
            local_ee_vel = self.convert_global_action_to_local(global_ee_vel)
            #if any term in local_ee_vel > 1 or < -1, break the step into smaller steps
            if np.any(local_ee_vel > 1*self.action_scale) or np.any(local_ee_vel < -1*self.action_scale):
                num_steps = int(np.ceil(np.max(np.abs(local_ee_vel/(1*self.action_scale)))))
                for j in range(num_steps):
                    ee_vel.append(local_ee_vel/num_steps)
            else:
                ee_vel.append(local_ee_vel)
        return ee_vel

class ResultMode(Enum):
    NO_SOLUTION = 0
    SUCCESS = 1
    NO_PATH = 2
