#TODO: Fix this file by subclassing

import pickle

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CallbackList
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import cv2
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
import random
import pandas as pd
import imageio
import copy
import math
def get_reachable_euclidean_grid(radius, resolution):
    num_bins = int(radius / resolution) * 2
    base_center = np.array([0, 0, 0.91])

    # Calculate the centers of the bins
    centers = np.zeros((num_bins, num_bins, num_bins, 3))
    for i in range(-num_bins // 2, num_bins // 2):
        for j in range(-num_bins // 2, num_bins // 2):
            for k in range(-num_bins // 2, num_bins // 2):
                # Calculate the center of the bin
                centers[i, j, k] = [(i + 0.5) * resolution, (j + 0.5) * resolution, (k + 0.5) * resolution]

    # Filter out the bins that are outside the semi-sphere or below the 45-degree line in the z direction
    valid_centers = []
    for i in range(-num_bins // 2, num_bins // 2):
        for j in range(-num_bins // 2, num_bins // 2):
            for k in range(-num_bins // 2, num_bins // 2):
                x, y, z = centers[i, j, k]
                if x ** 2 + y ** 2 + z ** 2 <= radius ** 2 and y < -0.7 and z > -0.05:  # and z <= np.tan(np.radians(45)) * np.sqrt(x**2 + y**2):
                    valid_centers.append((x + base_center[0], y + base_center[1], z + base_center[2]))

    # valid_centers = np.array(valid_centers)
    return valid_centers

def rand_direction_vector(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    # convert M to euler angles
    return M@[0,0,1]

    return M

def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points
def roundup(x):
    return math.ceil(x / 10.0) * 10

def rounddown(x):
    return math.floor(x / 10.0) * 10

def get_bin_from_orientation(orientation):
    offset = 1e-4
    orientation = orientation / np.linalg.norm(orientation)
    lat_angle = np.rad2deg(np.arcsin(orientation[2])) + offset
    lon_angle = np.rad2deg(np.arctan2(orientation[1], orientation[0])) + offset
    lat_angle_min = rounddown(lat_angle)
    lat_angle_max = roundup(lat_angle)
    lon_angle_min = rounddown(lon_angle)
    lon_angle_max = roundup(lon_angle)
    bin_key = (round((lat_angle_min + lat_angle_max) / 2), round((lon_angle_min + lon_angle_max) / 2))
    # if bin_key[0] not in between -85 and 85 set as 85 or -85
    # if bin_keyp[1] not in between -175 and 175 set as 175 or -175

    if bin_key[0] > 85:
        bin_key = (85, bin_key[1])
    elif bin_key[0] < -85:
        bin_key = (-85, bin_key[1])
    if bin_key[1] > 175:
        bin_key = (bin_key[0], 175)
    elif bin_key[1] < -175:
        bin_key = (bin_key[0], -175)
    return bin_key
class CustomTrainCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, or_bins, eval_freq, best_model_save_path, verbose=0):
     
        super(CustomTrainCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
       #self. = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        # local and global variables
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal

        # Sometimes, for event callback, it is useful
        # to havd access to the parent object
        self.parent = None  # type: Optional[BaseCallback]
        self._screens_buffer = []
        self._reward_dict = {}
        self._info_dict = {}
        self._rollouts = 0
        self._train_record_freq = 200

        self.or_bins = or_bins
        self.delta_pos_max = np.array([1, -0.675, 0])
        self.delta_pos_min = np.array([-1, -0.9525, -2])
        self.reachable_euclidean_grid = get_reachable_euclidean_grid(0.95, 0.05)

        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path





    def _init_callback(self) -> None:
        for i in range(self.training_env.num_envs):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point()
            self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                         point_branch_normal = current_branch_normal)


    def _sample_tree_and_point(self):
            #Sample orientation from key in or_bins
            while True:
                rand_vector = rand_direction_vector()
                orientation = get_bin_from_orientation(rand_vector)
                if len(self.or_bins[orientation]) == 0:
                    print("No trees in orientation", orientation)
                    continue
                tree_urdf, random_point, tree_orientation, scale = random.choice(self.or_bins[orientation])
                required_point_pos = random.choice(self.reachable_euclidean_grid)

                current_point_pos = random_point[0]
                current_branch_or = random_point[1]
                current_branch_normal = random_point[2]
                offset = np.random.uniform(-0.025, 0.025, 3)
                delta_tree_pos = np.array(required_point_pos) - np.array(current_point_pos) + offset
                final_point_pos = np.array(current_point_pos) + delta_tree_pos

                if (delta_tree_pos > self.delta_pos_max).any() or (delta_tree_pos < self.delta_pos_min).any():
                     # print("Invalid delta pos", delta_tree_pos, "required pos", required_point_pos, "current pos", current_point_pos, "offset", offset)
                     continue

                break

            return tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, delta_tree_pos, current_branch_normal

    def _update_tree_properties(self, infos):
        for i in range(len(infos)):
            if infos[i]["TimeLimit.truncated"] or infos[i]['is_success']:
                tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal\
                    = self._sample_tree_and_point()
                self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                            point_pos = final_point_pos, point_branch_or=current_branch_or,
                                            tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                             point_branch_normal=current_branch_normal)



    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        return True

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print("Rollout start")
        self._reward_dict["movement_reward"] = []
        self._reward_dict["distance_reward"] = []
        self._reward_dict["terminate_reward"] = []
        self._reward_dict["collision_acceptable_reward"] = []
        self._reward_dict["collision_unacceptable_reward"] = []
        self._reward_dict["slack_reward"] = []
        self._reward_dict["condition_number_reward"] = []
        self._reward_dict["velocity_reward"] = []
        self._reward_dict["perpendicular_orientation_reward"] = []
        self._reward_dict["pointing_orientation_reward"] = []
        self._info_dict["pointing_cosine_sim_error"] = []
        self._info_dict["perpendicular_cosine_sim_error"] = []
        self._info_dict["euclidean_error"] = []
        self._info_dict["is_success"] = []
        self._info_dict['velocity'] = []
        self._screens_buffer = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        infos = self.locals["infos"]
        for i in range(len(infos)):
            for key in self._reward_dict.keys():
                self._reward_dict[key].append(infos[i][key])
            if infos[i]["TimeLimit.truncated"] or infos[i]["is_success"]:
                for key in self._info_dict.keys():
                    self._info_dict[key].append(infos[i][key])

        self._update_tree_properties(infos)
        if self._rollouts % self._train_record_freq == 0:
            #grab screen
            self._screens_buffer.append(self._grab_screen_callback(self.locals, self.globals))

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.best_model_save_path is not None:
                os.makedirs(self.best_model_save_path, exist_ok=True)
            self.model.save(os.path.join(self.best_model_save_path, "current_model_{}".format(self.num_timesteps)))
            with open(os.path.join(self.best_model_save_path, "current_mean_std_{}.pkl".format(self.num_timesteps)), "wb") as f:
                pickle.dump((self.model.policy.running_mean_var_oflow_x, self.model.policy.running_mean_var_oflow_y), f)
        return True
    
    def _grab_screen_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        """
        Renders the environment in its current state, recording the screen in the captured `screens` list

        :param _locals: A dictionary containing all local variables of the callback's scope
        :param _globals: A dictionary containing all global variables of the callback's scope
        """
       
        screen = np.array(self.training_env.render())*255
        screen_copy = screen.reshape((screen.shape[0], screen.shape[1], 3)).astype(np.uint8)
        return screen_copy.transpose(2, 0, 1)
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print("Rollout end")
        #TODO: Make this a dictionary and iterate through dictionary to log
        for key in self._reward_dict.keys():
            # print("rollout/"+key, np.mean(self._reward_dict[key])   )
            self.logger.record("rollout/"+key, np.mean(self._reward_dict[key]))
        for key in self._info_dict.keys():
            self.logger.record("rollout/"+key, np.mean(self._info_dict[key]))
        if self._rollouts % self._train_record_freq == 0:
            self.logger.record(
                "rollout/video",
                Video(th.ByteTensor(np.array([self._screens_buffer])), fps=10),
                exclude=("stdout", "log", "json", "csv"),
            )
        self._rollouts += 1
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.

        """
        pass

class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        record_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        or_bins: Dict = None,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        if not isinstance(record_env, VecEnv):
            record_env = DummyVecEnv([lambda: record_env])

        self.eval_env = eval_env
        #Monitor, single env, do not vectorize
        self.record_env = record_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        #For video
        self._screens_buffer = []
        self._collisions_acceptable_buffer = []
        self._collisions_unacceptable_buffer = []
        self._reward_dict = {}
        self._info_dict = {}

        self.or_bins = or_bins
        self.delta_pos_max = np.array([1, -0.675, 0])
        self.delta_pos_min = np.array([-1, -0.9525, -2])
        self.reachable_euclidean_grid = get_reachable_euclidean_grid(0.95, 0.05)
        self.dataset = None

        # divide n_eval_episodes by n_envs
        self.current_index = [self.n_eval_episodes // self.eval_env.num_envs * i for i in range(self.eval_env.num_envs)]
        self.episode_counter = 0

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        # if not isinstance(self.training_env, type(self.eval_env)):
        #     Warning.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

        for i in range(self.eval_env.num_envs):

            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(i)
            self.eval_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos, point_branch_normal = current_branch_normal)
        for i in range(self.record_env.num_envs):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = random.choice(self.dataset)
            self.record_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos, point_branch_normal = current_branch_normal)

    def _sample_tree_and_point(self, idx):
        if self.dataset is None:
            self.dataset = self._make_dataset()
            print("Dataset made", len(self.dataset))
        print("Sampling for {} with id {}".format(idx, self.current_index[idx]))
        tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal = self.dataset[
            self.current_index[idx]]
        self.current_index[idx] = min(self.current_index[idx] + 1,
                                      self.n_eval_episodes // self.eval_env.num_envs * (idx + 1) - 1)
        return tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal

    def _make_dataset(self):
        points = fibonacci_sphere(samples=self.n_eval_episodes)
        # bins = lambda x: get_bin_from_orientation(x), points
        bins = [get_bin_from_orientation(x) for x in points]
        num_bins = len(set(bins))
        dataset = []
        for i in range(self.n_eval_episodes):
            while True:
                orientation = bins[i % num_bins]
                if len(self.or_bins[orientation]) == 0:
                    print("No trees in orientation", orientation)
                    i+=1
                    continue
                tree_urdf, random_point, tree_orientation, scale = random.choice(self.or_bins[orientation])
                required_point_pos = random.choice(self.reachable_euclidean_grid)

                current_point_pos = random_point[0]
                current_branch_or = random_point[1]
                current_branch_normal = random_point[2]
                offset = np.random.uniform(-0.025, 0.025, 3)
                delta_tree_pos = np.array(required_point_pos) - np.array(current_point_pos) + offset
                final_point_pos = np.array(current_point_pos) + delta_tree_pos
                # print(orientation, final_point_pos)
                if (delta_tree_pos > self.delta_pos_max).any() or (delta_tree_pos < self.delta_pos_min).any():
                    continue
                dataset.append((tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, delta_tree_pos, current_branch_normal))
                break
        return dataset

    def update_tree_properties(self, info, idx, name):
        if name == "eval" and (info["TimeLimit.truncated"] or info['is_success']):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(idx)
            self.eval_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale,
                                         tree_pos=tree_pos, point_branch_normal=current_branch_normal)


            self.episode_counter += 1
        elif name == "record" and info["TimeLimit.truncated"] or info['is_success']:
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = random.choice(self.dataset)
            self.record_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale,
                                         tree_pos=tree_pos, point_branch_normal = current_branch_normal)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """

        info = locals_["info"]
        if locals_["done"]: #Log at end of episode
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _log_rewards_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]
        #add to buffers

        for key in self._reward_dict.keys():
            self._reward_dict[key].append(infos[key])
        if infos["TimeLimit.truncated"] or infos['is_success']:
            for key in self._info_dict.keys():
                self._info_dict[key].append(infos[key])
        return True

    def _grab_screen_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        """
        Renders the environment in its current state, recording the screen in the captured `screens` list

        :param _locals: A dictionary containing all local variables of the callback's scope
        :param _globals: A dictionary containing all global variables of the callback's scope
        """
        episode_counts = _locals["episode_counts"][0]
        observation_info = self.record_env.get_attr("observation_info", 0)[0]




        if episode_counts == 0:
            render = np.array(self.record_env.render())*255
            render = cv2.resize(render, (512, 512), interpolation=cv2.INTER_NEAREST)
            rgb = observation_info["rgb"]*255
            rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)
            screen = np.concatenate((render, rgb), axis=1)
            screen_copy = screen.reshape((screen.shape[0], screen.shape[1], 3)).astype(np.uint8)
            # screen_copy = cv2.resize(screen_copy, (1124, 768), interpolation=cv2.INTER_NEAREST)
            screen_copy = cv2.putText(screen_copy, "Reward: "+str(_locals['reward']), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255,0,0), 2, cv2.LINE_AA)
            screen_copy = cv2.putText(screen_copy, "Action: "+" ".join(str(x) for x in _locals['actions']), (0,110), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
            screen_copy = cv2.putText(screen_copy, "Current: "+str(observation_info['achieved_pos']), (0,140), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255,0,0), 2, cv2.LINE_AA)
            screen_copy = cv2.putText(screen_copy, "Goal: "+str(observation_info['desired_pos']), (0,170), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,0,0), 2, cv2.LINE_AA)
            screen_copy = cv2.putText(screen_copy, "Orientation Perpendicular: " +str(observation_info['perpendicular_cosine_sim']), (0,200), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
            screen_copy = cv2.putText(screen_copy, "Orientation Pointing: " + str(
                observation_info['pointing_cosine_sim']), (0, 230),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 0, 0), 2, cv2.LINE_AA)  # str(_locals['actions'])

            screen_copy = cv2.putText(screen_copy, "Distance: "+" ".join(str(observation_info["target_distance"])), (0,260), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
            self._screens_buffer.append(screen_copy.transpose(2, 0, 1))

        info = _locals["info"]
        i = _locals['i']
        self.update_tree_properties(info, i, "record")
        # print("Saving screen")

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        i = _locals['i']
        self._collisions_acceptable_buffer.append(self.eval_env.get_attr("collisions_acceptable", i)[0])
        self._collisions_unacceptable_buffer.append(self.eval_env.get_attr("collisions_unacceptable", i)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        # self._grab_screen_callback(_locals, _globals)
        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)

        info = _locals["info"]
        i = _locals['i']
        self.update_tree_properties(info, i, "eval")


        #change tree jere

    def _init_dicts(self):
        self._reward_dict["movement_reward"] = []
        self._reward_dict["distance_reward"] = []
        self._reward_dict["terminate_reward"] = []
        self._reward_dict["collision_acceptable_reward"] = []
        self._reward_dict["collision_unacceptable_reward"] = []
        self._reward_dict["slack_reward"] = []
        self._reward_dict["condition_number_reward"] = []
        self._reward_dict["velocity_reward"] = []
        self._reward_dict["perpendicular_orientation_reward"] = []
        self._reward_dict["pointing_orientation_reward"] = []
        self._info_dict["pointing_cosine_sim_error"] = []
        self._info_dict["perpendicular_cosine_sim_error"] = []
        self._info_dict["euclidean_error"] = []
        self._info_dict['velocity'] = []

    def _evalute_policy(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=[self._master_callback, self._grab_screen_callback],
        )
        return episode_rewards, episode_lengths

    def _record_policy(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        _, _ = evaluate_policy(
            self.model,
            self.record_env,
            n_eval_episodes=1,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=False,
            warn=self.warn,
            callback=self._grab_screen_callback,
        )

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self._screens_buffer = []
            self._collisions_buffer = []
            self._init_dicts()

            # Evaluate policy
            print("Evaluating")
            import time
            start = time.time()
            episode_rewards, episode_lengths = self._evalute_policy(self.locals, self.globals)
            self._record_policy(self.locals, self.globals)
            end = time.time()
            print("Evaluation took: ", end-start)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_collisions_acceptable = np.mean(self._collisions_acceptable_buffer)
            mean_collisions_unacceptable = np.mean(self._collisions_unacceptable_buffer)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record(
                    "eval/video",
                    Video(th.ByteTensor(np.array([self._screens_buffer])), fps=10),
                    exclude=("stdout", "log", "json", "csv"),
                )
            self.logger.record("eval/collisions_acceptable", mean_collisions_acceptable)
            self.logger.record("eval/collisions_unacceptable", mean_collisions_unacceptable)
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            for key in self._reward_dict.keys():
                self.logger.record("eval/"+key, np.mean(self._reward_dict[key]))
            for key in self._info_dict.keys():
                self.logger.record("eval/"+key, np.mean(self._info_dict[key]))


            #Save current model
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "current_model"))
                with open(os.path.join(self.best_model_save_path, "current_mean_std.pkl"), "wb") as f:
                    pickle.dump((self.model.policy.running_mean_var_oflow_x, self.model.policy.running_mean_var_oflow_y), f)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    with open(os.path.join(self.best_model_save_path, "best_mean_std.pkl"), "wb") as f:
                        pickle.dump((self.model.policy.running_mean_var_oflow_x, self.model.policy.running_mean_var_oflow_y), f)


                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
            print("Done evaluating")
            self.episode_counter = 0
            #reset current index
            self.current_index = [self.n_eval_episodes // self.eval_env.num_envs * i for i in range(self.eval_env.num_envs)]
        return continue_training

class CustomResultCallback(EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        or_bins: Dict = None,
        dataset: List = None,
        save_video: bool = False,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.save_video = save_video
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        #Monitor, single env, do not vectorize
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []


        self._reward_dict_temp = {}
        self._info_dict_temp = {}
        self._reward_dict_list = []
        self._info_dict_list = []

        self._terminal_dict = {}


        self.or_bins = or_bins
        self.delta_pos_max = np.array([1, -0.675, 0])
        self.delta_pos_min = np.array([-1, -0.9525, -2])
        self.reachable_euclidean_grid = get_reachable_euclidean_grid(0.95, 0.05)

        self.episode_counter = 0
        self.dataset = dataset
        self.num_points_per_or = 5

        #divide n_eval_episodes by n_envs
        self.current_index = None#[(self.n_eval_episodes*self.num_points_per_or)//self.eval_env.num_envs*i for i in range(self.eval_env.num_envs)]

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        # if not isinstance(self.training_env, type(self.eval_env)):
        #     Warning.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
        # Create folders if needed

        for i in range(self.eval_env.num_envs):

            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(i)
            self.eval_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos, point_branch_normal = current_branch_normal)
        self.current_index = [(len(self.dataset)) // self.eval_env.num_envs * i for i in
                              range(self.eval_env.num_envs)]

    def _sample_tree_and_point(self, idx):
        if self.dataset is None:
            self.dataset = self._make_dataset()
            self.current_index = [(len(self.dataset)) // self.eval_env.num_envs * i for i in
                                  range(self.eval_env.num_envs)]
            print("Dataset made", len(self.dataset))
        # print("Sampling for {} with id {}".format(idx, self.current_index[idx]))
        tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal = self.dataset[self.current_index[idx]]
        self.current_index[idx] = min(self.current_index[idx]+1, len(self.dataset)//self.eval_env.num_envs*(idx+1)-1)
        return tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal

    def _make_dataset(self):
        # Make this dataset to sample all orientations, and 10 random required_point_pos for each orientation
        lat_range = (-85, 95)
        lon_range = (-175, 185)
        #
        # # Number of bins
        # # num_bins = 648
        # # num_bins_per_axis = int(np.sqrt(num_bins))
        # num_bins_lat = 18
        # num_bins_lon = 36
        # lat_step = int(lat_range[1] - lat_range[0]) // num_bins_lat
        # lon_step = int(lon_range[1] - lon_range[0]) // num_bins_lon
        #

        # if eval checkpoints
            # points = fibonacci_sphere(samples=self.n_eval_episodes)
            # or_list = [get_bin_from_orientation(x) for x in points]
            # num_bins = len(set(or_list))
        # Create the bins
        lat_step = 10
        lon_step = 10
        lat_bins = np.arange(lat_range[0], lat_range[1], lat_step, dtype=int)
        lon_bins = np.arange(lon_range[0], lon_range[1], lon_step, dtype=int)
        #Make a grid
        lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)
        or_list = list(zip(lat_grid.flatten(), lon_grid.flatten()))
        print("Orientation list", or_list)
        num_bins = len(or_list)


        dataset = []
        num_points_per_or = self.num_points_per_or
        for i in range(num_bins):
            for j in range(num_points_per_or):
                while True:
                    orientation = or_list[i % num_bins]
                    if len(self.or_bins[orientation]) == 0:
                        print("No trees in orientation", orientation)
                        continue
                    tree_urdf, random_point, tree_orientation, scale = random.choice(self.or_bins[orientation])
                    required_point_pos = random.choice(self.reachable_euclidean_grid)

                    current_point_pos = random_point[0]
                    current_branch_or = random_point[1]
                    current_branch_normal = random_point[2]
                    offset = np.random.uniform(-0.025, 0.025, 3)
                    delta_tree_pos = np.array(required_point_pos) - np.array(current_point_pos) + offset
                    final_point_pos = np.array(current_point_pos) + delta_tree_pos
                    # print(orientation, final_point_pos)
                    if (delta_tree_pos > self.delta_pos_max).any() or (delta_tree_pos < self.delta_pos_min).any():
                        continue
                    dataset.append((tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, delta_tree_pos, current_branch_normal))
                    break
        return dataset

    def update_tree_properties(self, info, idx, name):
        if name == "eval" and (info["TimeLimit.truncated"] or info['is_success']):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(idx)
            self.eval_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale,
                                         tree_pos=tree_pos, point_branch_normal=current_branch_normal)


            self.episode_counter += 1

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """

        info = locals_["info"]
        if locals_["done"]: #Log at end of episode
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _log_rewards_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]
        i = locals_['i']
        #add to buffers

        for key in self._reward_dict_list[i].keys():
            self._reward_dict_list[i][key].append(infos[key])

        return True

    def _log_final_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]
        i = locals_['i']
        if infos["TimeLimit.truncated"] or infos['is_success']:
            print("Logging final metrics")
            for key in self._info_dict_list[i].keys():
                self._info_dict_list[i][key].append(infos[key])
            self._terminal_dict["init_distance"].append(self.eval_env.get_attr("init_distance", i)[0])
            self._terminal_dict["init_perp_cosine_sim"].append(self.eval_env.get_attr("init_perp_cosine_sim", i)[0])
            self._terminal_dict["init_point_cosine_sim"].append(self.eval_env.get_attr("init_point_cosine_sim", i)[0])
            self._terminal_dict["init_angular_error"].append(self.eval_env.get_attr("init_angular_error", i)[0])
            branch_loc = self.eval_env.get_attr("tree_goal_pos", i)[0]
            branch_or = self.eval_env.get_attr("tree_goal_or", i)[0]
            self._terminal_dict["pointx"].append(branch_loc[0])
            self._terminal_dict["pointy"].append(branch_loc[1])
            self._terminal_dict["pointz"].append(branch_loc[2])
            self._terminal_dict["or_x"].append(branch_or[0])
            self._terminal_dict["or_y"].append(branch_or[1])
            self._terminal_dict["or_z"].append(branch_or[2])


            for key, value in self._reward_dict_list[i].items():
                self._episode_info[key].append(np.mean(value))
            for key, value in self._info_dict_list[i].items():
                self._episode_info[key].append(value[0])

            self._episode_info["acceptable_collision"].append(np.mean(self._collisions_acceptable_buffer[i]))
            self._episode_info["unacceptable_collision"].append(np.mean(self._collisions_unacceptable_buffer[i]))
            self._episode_info['is_success'].append(infos['is_success'])
            #Save video

            observation_info = self.eval_env.get_attr("observation_info", i)[0]
            tree_goal_pos = self.eval_env.get_attr("tree_goal_pos", i)[0]
            if self.save_video:
                imageio.mimsave("results/{}_{}.gif".format(observation_info["desired_pos"], tree_goal_pos), self._screens_buffer[i])
            self._screens_buffer[i] = []
            self._collisions_acceptable_buffer[i] = []
            self._collisions_unacceptable_buffer[i] = []
            for key in self._reward_dict_list[i].keys():
                self._reward_dict_list[i][key] = []
            for key in self._info_dict_list[i].keys():
                self._info_dict_list[i][key] = []

    def _grab_screen_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        """
        Renders the environment in its current state, recording the screen in the captured `screens` list

        :param _locals: A dictionary containing all local variables of the callback's scope
        :param _globals: A dictionary containing all global variables of the callback's scope
        """
        i = _locals['i']
        observation_info = self.eval_env.get_attr("observation_info", i)[0]
        render = np.array(self.eval_env.env_method("render", indices=i))[0]*255
        render = cv2.resize(render, (512, 512), interpolation=cv2.INTER_NEAREST)
        rgb = observation_info["rgb"]*255
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)
        screen = np.concatenate((render, rgb), axis=1)
        screen_copy = screen.reshape((screen.shape[0], screen.shape[1], 3)).astype(np.uint8)
        # screen_copy = cv2.resize(screen_copy, (1124, 768), interpolation=cv2.INTER_NEAREST)
        screen_copy = cv2.putText(screen_copy, "Reward: "+str(_locals['reward']), (0,80), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,0,0), 2, cv2.LINE_AA)
        screen_copy = cv2.putText(screen_copy, "Action: "+" ".join(str(x) for x in _locals['actions']), (0,110), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
        screen_copy = cv2.putText(screen_copy, "Current: "+str(observation_info['achieved_pos']), (0,140), cv2.FONT_HERSHEY_SIMPLEX,
            .5, (255,0,0), 2, cv2.LINE_AA)
        screen_copy = cv2.putText(screen_copy, "Goal: "+str(observation_info['desired_pos']), (0,170), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,0,0), 2, cv2.LINE_AA)
        screen_copy = cv2.putText(screen_copy, "Orientation Perpendicular: " +str(observation_info['perpendicular_cosine_sim']), (0,200), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
        screen_copy = cv2.putText(screen_copy, "Orientation Pointing: " + str(
            observation_info['pointing_cosine_sim']), (0, 230),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 0, 0), 2, cv2.LINE_AA)  # str(_locals['actions'])

        screen_copy = cv2.putText(screen_copy, "Distance: "+" ".join(str(observation_info["target_distance"])), (0,260), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255,0,0), 2, cv2.LINE_AA)
        #Add success label
        # screen_copy = cv2.cvtColor(screen_copy, cv2.COLOR_BGR2RGB)
        self._screens_buffer[i].append((screen_copy).astype(np.uint8))
    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        i = _locals['i']
        self._collisions_acceptable_buffer[i].append(self.eval_env.get_attr("collisions_acceptable", i)[0])
        self._collisions_unacceptable_buffer[i].append(self.eval_env.get_attr("collisions_unacceptable", i)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        # self._grab_screen_callback(_locals, _globals)
        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)
        if self.save_video:
            self._grab_screen_callback(_locals, _globals)
        self._log_final_metrics(_locals, _globals)

        info = _locals["info"]
        i = _locals['i']
        self.update_tree_properties(info, i, "eval")


        #change tree jere

    def _init_dicts(self):
        _reward_dict = {}
        _reward_dict["movement_reward"] = []
        _reward_dict["distance_reward"] = []
        _reward_dict["terminate_reward"] = []
        _reward_dict["collision_acceptable_reward"] = []
        _reward_dict["collision_unacceptable_reward"] = []
        _reward_dict["slack_reward"] = []
        _reward_dict["condition_number_reward"] = []
        _reward_dict["velocity_reward"] = []
        _reward_dict["perpendicular_orientation_reward"] = []
        _reward_dict["pointing_orientation_reward"] = []

        self._reward_dict_list = [copy.deepcopy(_reward_dict) for i in range(self.eval_env.num_envs)]

        _info_dict = {}
        _info_dict["pointing_cosine_sim_error"] = []
        _info_dict["perpendicular_cosine_sim_error"] = []
        _info_dict["euclidean_error"] = []
        _info_dict['velocity'] = []

        self._info_dict_list = [copy.deepcopy(_info_dict) for i in range(self.eval_env.num_envs)]

        self._terminal_dict["init_distance"] = []
        self._terminal_dict["init_perp_cosine_sim"] = []
        self._terminal_dict["init_point_cosine_sim"] = []
        self._terminal_dict["init_angular_error"] = []
        self._terminal_dict["pointx"] = []
        self._terminal_dict["pointy"] = []
        self._terminal_dict["pointz"] = []
        self._terminal_dict["or_x"] = []
        self._terminal_dict["or_y"] = []
        self._terminal_dict["or_z"] = []

        self._episode_info = {}
        self._episode_info["is_success"] = []
        self._episode_info["acceptable_collision"] = []
        self._episode_info["unacceptable_collision"] = []

        for i in _reward_dict.keys():
            self._episode_info[i] = []
        for i in _info_dict.keys():
            self._episode_info[i] = []

        self._collisions_acceptable_buffer = [[] for i in range(self.eval_env.num_envs)]
        self._collisions_unacceptable_buffer = [[] for i in range(self.eval_env.num_envs)]
        # For video
        self._screens_buffer = [[] for i in range(self.eval_env.num_envs)]
    def get_results(self):
        # Reset success rate buffer
        self._is_success_buffer = []
        self._screens_buffer = [[] for i in range(self.eval_env.num_envs)]
        self._collisions_buffer = []
        self._init_dicts()

        # Evaluate policy
        print("Evaluating")
        import time
        start = time.time()
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=len(self.dataset),
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._master_callback,
        )

        end = time.time()
        print("Evaluation took: ", end-start)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )
        # print(self._episode_info)
        # print(self._terminal_dict)
        # for i in self._episode_info.keys():
        #     print(i, len(self._episode_info[i]))
        # for i in self._terminal_dict.keys():
        #     print(i, len(self._terminal_dict[i]))
        episode_info_df = pd.DataFrame(self._episode_info)
        terminal_info_df = pd.DataFrame(self._terminal_dict)
        save_df = pd.concat([episode_info_df, terminal_info_df], axis=1)
        save_df.to_csv("episode_info.csv", mode='a')

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        self.episode_counter = 0
        return mean_reward, std_reward


