
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


class RRTCallback(EventCallback):
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
        n_eval_episodes: int = 5,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        or_bins: Dict = None,
        save_video: bool = False,
        dataset = None
    ):
        super().__init__(None, verbose=verbose)

        self.n_eval_episodes = n_eval_episodes
        self.render = render
        self.warn = warn
        self.save_video = save_video
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        #Monitor, single env, do not vectorize
        self.or_bins = or_bins
        self.delta_pos_max = np.array([1, -0.675, 0])
        self.delta_pos_min = np.array([-1, -0.9525, -2])
        self.reachable_euclidean_grid = get_reachable_euclidean_grid(0.95, 0.05)

        self.episode_counter = 0
        self.num_points_per_or = 5
        self.dataset = dataset

        #divide n_eval_episodes by n_envs
        self.current_index = [(self.n_eval_episodes*self.num_points_per_or)//self.eval_env.num_envs*i for i in range(self.eval_env.num_envs)]

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
                                     tree_pos=tree_pos, point_branch_normal=current_branch_normal)
        self.current_index = [(self.n_eval_episodes * self.num_points_per_or) // self.eval_env.num_envs * i for i in
                              range(self.eval_env.num_envs)]

    def _sample_tree_and_point(self, idx):
        if self.dataset is None:
            self.dataset = self._make_dataset()
            #Write the dataset to a file
            with open("rrt_dataset.pkl", "wb") as f:
                pickle.dump(self.dataset, f)
            print("Dataset made", len(self.dataset))
        # print("Sampling for {} with id {}".format(idx, self.current_index[idx]))
        tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal = self.dataset[self.current_index[idx]]
        # if self.current_index[idx] < len(self.dataset)//self.eval_env.num_envs*(idx+1)-1:
        self.current_index[idx] = min(self.current_index[idx]+1, len(self.dataset)//self.eval_env.num_envs*(idx+1)-1)
        return tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal

    def _make_dataset(self):
        # Make this dataset to sample all orientations, and 10 random required_point_pos for each orientation
        lat_range = (-85, 95)
        lon_range = (-175, 185)

        lat_step = 10
        lon_step = 10
        lat_bins = np.arange(lat_range[0], lat_range[1], lat_step, dtype=int)
        lon_bins = np.arange(lon_range[0], lon_range[1], lon_step, dtype=int)
        #Make a grid
        lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)
        or_list = list(zip(lat_grid.flatten(), lon_grid.flatten()))
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

        print("Dataset made", len(dataset))
        return dataset

    def update_tree_properties(self, idx):
        tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
            = self._sample_tree_and_point(idx)
        self.eval_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos, point_branch_normal=current_branch_normal)


        self.episode_counter += 1


