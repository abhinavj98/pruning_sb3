# TODO: Fix this file by subclassing

import copy
import math
import os
import pickle
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import cv2
import gymnasium as gym

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class PruningSetGoalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PruningSetGoalCallback, self).__init__(verbose)

    @abstractmethod
    def _sample_tree_and_point(self, idx):
        pass

    @abstractmethod
    def _update_tree_properties(self):
        pass

    @staticmethod
    def get_reachable_euclidean_grid(radius, resolution):
        num_bins = round(radius / resolution) * 2
        base_center = np.array([0, 0, 0.91])
        # Create a 3D grid of indices
        i, j, k = np.mgrid[-num_bins // 2:num_bins // 2, -num_bins // 2:num_bins // 2, -num_bins // 2:num_bins // 2]

        # Scale and shift the grid to get the centers of the bins
        centers = np.stack([(i + 0.5) * resolution, (j + 0.5) * resolution, (k + 0.5) * resolution],
                           axis=-1)

        # Create a mask for the valid centers
        mask = (centers[..., 0] ** 2 + centers[..., 1] ** 2 + centers[..., 2] ** 2 <= radius ** 2) & (
                centers[..., 1] < -0.7) & (centers[..., 2] > -0.05)

        # Apply the mask to the centers array to get the valid centers
        valid_centers = centers[mask] + base_center
        valid_centers = list(map(tuple, valid_centers))
        return valid_centers

    @staticmethod
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
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
        )

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)

        # Rotate the vector
        return M @ [0, 0, 1]

    @staticmethod
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

    @staticmethod
    def roundup(x):
        return math.ceil(x / 10.0) * 10

    @staticmethod
    def rounddown(x):
        return math.floor(x / 10.0) * 10

    @staticmethod
    def get_bin_from_orientation(orientation):
        offset = 1e-4
        orientation = orientation / np.linalg.norm(orientation)
        lat_angle = np.rad2deg(np.arcsin(orientation[2])) + offset
        lon_angle = np.rad2deg(np.arctan2(orientation[1], orientation[0])) + offset

        lat_angle_min = PruningSetGoalCallback.rounddown(lat_angle)
        lat_angle_max = PruningSetGoalCallback.roundup(lat_angle)
        lon_angle_min = PruningSetGoalCallback.rounddown(lon_angle)
        lon_angle_max = PruningSetGoalCallback.roundup(lon_angle)

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

    def maybe_sample_point(self, orientation):
        if len(self.or_bins[orientation]) == 0:
            if self.verbose > 1:
                print(f"DEBUG: No trees in orientation {orientation}")
        tree_urdf, random_point, tree_orientation, scale, collision_meshes = random.choice(self.or_bins[orientation]) #TODO: Also return collision_objects
        current_point_pos, current_branch_or, current_branch_normal, _ = random_point
        required_point_pos = random.choice(self.reachable_euclidean_grid)

        offset = np.random.uniform(-0.025, 0.025, 3)
        delta_tree_pos = np.array(required_point_pos) - np.array(current_point_pos) + offset
        final_point_pos = np.array(current_point_pos) + delta_tree_pos
        # print(orientation, final_point_pos)
        if (delta_tree_pos > self.delta_pos_max).any() or (delta_tree_pos < self.delta_pos_min).any():
            if self.verbose > 1:
                print(
                    f"DEBUG: Invalid delta pos {delta_tree_pos}, required pos {required_point_pos}, current pos {current_point_pos}, offset {offset}")
            return False, None
        return True, (tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, delta_tree_pos, current_branch_normal, collision_meshes)


class EveryNRollouts(EventCallback):
    """
    Trigger a callback every ``n_rollouts``

    :param n_rollouts: Number of rollouts between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(self, n_rollouts: int, callback: BaseCallback):
        super().__init__(callback)
        self.n_rollouts = n_rollouts
        self.num_rollouts = 0

    def _on_rollout_end(self) -> None:
        if (self.num_rollouts % self.n_rollouts) == 0:
            self.callback._on_rollout_end()
        self.num_rollouts += 1

    def _on_step(self) -> bool:
        if (self.num_rollouts % self.n_rollouts) == 0:
            return self._on_event()
        return True


class PruningLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PruningLogCallback, self).__init__(verbose)
        self._reward_dict = {}
        self._info_dict = {}
        self._collisions_acceptable_buffer = []
        self._collisions_unacceptable_buffer = []

    def _init_log_storage(self):
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
        self._collisions_acceptable_buffer = []
        self._collisions_unacceptable_buffer = []

    def _log_infos(self):
        infos = self.locals["infos"]
        for i in range(len(infos)):
            for key in self._reward_dict.keys():
                self._reward_dict[key].append(infos[i][key])
            if infos[i]["TimeLimit.truncated"] or infos[i]["is_success"]:
                for key in self._info_dict.keys():
                    self._info_dict[key].append(infos[i][key])

    def _log_collisions(self) -> None:
        self._collisions_acceptable_buffer.extend(self.training_env.get_attr("collisions_acceptable"))
        self._collisions_unacceptable_buffer.extend(self.training_env.get_attr("collisions_unacceptable"))

    def _on_rollout_start(self) -> None:
        if self.verbose > 0:
            print("INFO: Rollout start")
        self._init_log_storage()

    def _on_step(self) -> bool:
        self._log_infos()
        self._log_collisions()
        return True

    def _on_rollout_end(self) -> None:
        if self.verbose > 0:
            print("INFO: Rollout end")
            print("INFO: Success rate", np.mean(self._info_dict["is_success"]))
        self._log_to_tensorboard()

    def _log_to_tensorboard(self):
        for key in self._reward_dict.keys():
            self.logger.record("rollout/" + key, np.mean(self._reward_dict[key]))
        for key in self._info_dict.keys():
            self.logger.record("rollout/" + key, np.mean(self._info_dict[key]))
        self.logger.record("rollout/collisions_acceptable", np.mean(self._collisions_acceptable_buffer))
        self.logger.record("rollout/collisions_unacceptable", np.mean(self._collisions_unacceptable_buffer))

