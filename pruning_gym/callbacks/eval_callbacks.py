import copy
import os
from typing import Dict, Any, Union, Optional
import pickle
import cv2
import imageio
import numpy as np
import pandas as pd
from pruning_sb3.pruning_gym.callbacks.callbacks import PruningSetGoalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


class PruningEvalSetGoalCallback(PruningSetGoalCallback):
    def __init__(self, or_bins, type, dataset, num_orientations, num_points_per_or, verbose=0):
        super(PruningEvalSetGoalCallback, self).__init__(verbose)
        self.or_bins = or_bins
        self.dataset = dataset
        self.delta_pos_max = np.array([1, -0.675, 0])
        self.delta_pos_min = np.array([-1, -0.9525, -2])
        self.reachable_euclidean_grid = self.get_reachable_euclidean_grid(0.95, 0.05)
        self.type = type
        self.num_orientations = num_orientations
        self.num_points_per_or = num_points_per_or

    def get_polar_grid(self, resolution):
        lat_range = (-85, 95)
        lon_range = (-175, 185)
        lat_step = resolution
        lon_step = resolution

        lat_bins = np.arange(lat_range[0], lat_range[1], lat_step, dtype=int)
        lon_bins = np.arange(lon_range[0], lon_range[1], lon_step, dtype=int)

        # Make a grid
        lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)
        or_list = list(zip(lat_grid.flatten(), lon_grid.flatten()))
        return or_list

    def make_analysis_dataset(self):
        if self.verbose > 0:
            print("INFO: Making analysis dataset")
        or_list = self.get_polar_grid(10)
        if self.verbose > 1:
            print("DEBUG: Orientation List: ", or_list)

        num_bins = len(or_list)
        num_points_per_or = self.num_points_per_or

        dataset = []
        for i in range(num_bins):
            for j in range(num_points_per_or):
                point_sampled = False
                while not point_sampled:
                    orientation = or_list[i % num_bins]
                    point_sampled, point = self.maybe_sample_point(orientation)
                dataset.append(point)

        if self.verbose > 1:
            print("DEBUG: Length of dataset: ", len(dataset))
        return dataset

    def make_uniform_dataset(self):
        if self.verbose > 0:
            print("INFO: Making uniform dataset")
        num_points = self.num_orientations
        orientations = self.fibonacci_sphere(samples=num_points)
        print("Orientations: ", len(orientations))
        or_list = [self.get_bin_from_orientation(x) for x in orientations]
        num_bins = len(or_list)
        num_points_per_or = self.num_points_per_or

        dataset = []
        for i in range(num_bins):
            for j in range(num_points_per_or):
                point_sampled = False
                orientation = or_list[i % num_bins]

                while not point_sampled:
                    point_sampled, point = self.maybe_sample_point(orientation)
                dataset.append(point)

        if self.verbose > 1:
            print("DEBUG: Length of dataset: ", len(dataset), self.num_orientations*self.num_points_per_or)
        return dataset

    def _init_callback(self) -> None:
        if self.dataset is not None:
            self.current_index = [(len(self.dataset)) // self.training_env.num_envs * i for i in
                                  range(self.training_env.num_envs)]
        else:
            self.current_index = None
        for i in range(self.training_env.num_envs):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(i)
            self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                         point_branch_normal=current_branch_normal)

            self.current_index = [(len(self.dataset)) // self.training_env.num_envs * i for i in
                                  range(self.training_env.num_envs)]


    def make_dataset(self):
        if self.type == "uniform":
            self.dataset = self.make_uniform_dataset()
            if self.verbose > 0:
                print("INFO: Dataset made", len(self.dataset))
        elif self.type == "analysis":
            self.dataset = self.make_analysis_dataset()
            if self.verbose > 0:
                print("INFO: Dataset made", len(self.dataset))
    def _sample_tree_and_point(self, idx):
        if self.verbose > 0:
            print("DEBUG: Sampling tree and point")

        if self.dataset is None:
            self.make_dataset()
            self.current_index = [(len(self.dataset)) // self.training_env.num_envs * i for i in
                                  range(self.training_env.num_envs)]

        with open(f"{self.type}_dataset.pkl", "wb") as f:
            pickle.dump(self.dataset, f)

        if self.verbose > 1:
            print("DEBUG: Sampling for {} with id {}".format(idx, self.current_index[idx]))

        point = self.dataset[self.current_index[idx]]
        self.current_index[idx] = min(self.current_index[idx] + 1,
                                      len(self.dataset) // self.training_env.num_envs * (idx + 1) - 1)

        return point

    def _update_tree_properties(self, _locals: Dict[str, Any], _globals: Dict[str, Any]):
        info = _locals["info"]
        i = _locals['i']
        if info["TimeLimit.truncated"] or info['is_success']:
            if self.verbose > 1:
                print(f"DEBUG: Updating tree in env {i} via callback")
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(i)
            self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                         point_branch_normal=current_branch_normal)

    def _on_step(self, _locals, _globals) -> bool:
        self._update_tree_properties(_locals, _globals)
        return True


class PruningEvalRecordEnvCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PruningEvalRecordEnvCallback, self).__init__(verbose)
        self._screens_buffer = None

    def _init_callback(self) -> None:
        self._screens_buffer = [[] for i in range(self.training_env.num_envs)]

    def reset_buffer(self, i):
        self._screens_buffer[i] = []

    def _grab_screen(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        i = _locals['i']
        observation_info = self.training_env.get_attr("observation_info", i)[0]
        render = np.array(self.training_env.env_method("render", indices=i))[0] * 255
        render = cv2.resize(render, (512, 512), interpolation=cv2.INTER_NEAREST)
        rgb = observation_info["rgb"] * 255
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)
        screen = np.concatenate((render, rgb), axis=1)
        screen = screen.reshape((screen.shape[0], screen.shape[1], 3)).astype(np.uint8)
        screen_copy = self.add_info_on_image(screen, _locals, observation_info)

        self._screens_buffer[i].append(screen_copy.astype(np.uint8))

    def add_info_on_image(self, screen_copy, _locals, observation_info):
        # screen_copy = cv2.resize(screen_copy, (1124, 768), interpolation=cv2.INTER_NEAREST)
        screen_copy = cv2.putText(screen_copy, "Reward: " + str(_locals['reward']), (0, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 0, 0), 2, cv2.LINE_AA)
        screen_copy = cv2.putText(screen_copy, "Action: " + " ".join(str(x) for x in _locals['actions']), (0, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 0, 0), 2, cv2.LINE_AA)  # str(_locals['actions'])
        screen_copy = cv2.putText(screen_copy, "Current: " + str(observation_info['achieved_pos']), (0, 140),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  .5, (255, 0, 0), 2, cv2.LINE_AA)
        screen_copy = cv2.putText(screen_copy, "Goal: " + str(observation_info['desired_pos']), (0, 170),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 0, 0), 2, cv2.LINE_AA)
        screen_copy = cv2.putText(screen_copy,
                                  "Orientation Perpendicular: " + str(observation_info['perpendicular_cosine_sim']),
                                  (0, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 0, 0), 2, cv2.LINE_AA)  # str(_locals['actions'])
        screen_copy = cv2.putText(screen_copy, "Orientation Pointing: " + str(
            observation_info['pointing_cosine_sim']), (0, 230),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 0, 0), 2, cv2.LINE_AA)  # str(_locals['actions'])

        screen_copy = cv2.putText(screen_copy, "Distance: " + " ".join(str(observation_info["target_distance"])),
                                  (0, 260), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 0, 0), 2, cv2.LINE_AA)
        return screen_copy

    def _on_step(self, _locals, _globals) -> bool:
        if self.verbose > 1:
            print("DEBUG: Recording video")
        self._grab_screen(_locals, _globals)
        self.maybe_save_video(_locals, _globals)
        return True

    def maybe_save_video(self, locals_, globals_):
        infos = locals_["info"]
        i = locals_['i']
        observation_info = self.training_env.get_attr("observation_info", i)[0]
        tree_goal_pos = self.training_env.get_attr("tree_goal_pos", i)[0]
        if infos["TimeLimit.truncated"] or infos['is_success']:
            if self.verbose > 0:
                print("INFO: Saving video")
            imageio.mimsave("results/{}_{}.gif".format(observation_info["desired_pos"], tree_goal_pos),
                            self._screens_buffer[i])
            self.reset_buffer(i)


class PruningLogResultCallback(BaseCallback):
    def __init__(
            self,
            log_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.deterministic = deterministic
        self.render = render
        self.log_path = log_path

    def _init_callback(self) -> None:
        self._init_buffers()

    def _init_buffers(self):
        _reward_dict = {"movement_reward": [], "distance_reward": [], "terminate_reward": [],
                        "collision_acceptable_reward": [], "collision_unacceptable_reward": [], "slack_reward": [],
                        "condition_number_reward": [], "velocity_reward": [], "perpendicular_orientation_reward": [],
                        "pointing_orientation_reward": []}

        self._reward_dict_list = [copy.deepcopy(_reward_dict) for i in range(self.training_env.num_envs)]

        _info_dict = {"pointing_cosine_sim_error": [], "perpendicular_cosine_sim_error": [], "euclidean_error": [],
                      'velocity': [], "time": []}

        self._info_dict_list = [copy.deepcopy(_info_dict) for i in range(self.training_env.num_envs)]

        self._terminal_dict = {"init_distance": [], "init_perp_cosine_sim": [], "init_point_cosine_sim": [],
                               "init_angular_error": [], "pointx": [], "pointy": [], "pointz": [], "or_x": [],
                               "or_y": [], "or_z": []}

        self._episode_info = {"is_success": [], "acceptable_collision": [], "unacceptable_collision": []}

        for i in _reward_dict.keys():
            self._episode_info[i] = []
        for i in _info_dict.keys():
            self._episode_info[i] = []

        self._collisions_acceptable_buffer = [[] for i in range(self.training_env.num_envs)]
        self._collisions_unacceptable_buffer = [[] for i in range(self.training_env.num_envs)]

    def _log_rewards(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]
        i = locals_['i']
        for key in self._reward_dict_list[i].keys():
            self._reward_dict_list[i][key].append(infos[key])

        return True

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        i = _locals['i']
        self._collisions_acceptable_buffer[i].append(self.training_env.get_attr("collisions_acceptable", i)[0])
        self._collisions_unacceptable_buffer[i].append(self.training_env.get_attr("collisions_unacceptable", i)[0])

    def reset_buffers(self, i):
        self._collisions_acceptable_buffer[i] = []
        self._collisions_unacceptable_buffer[i] = []
        for key in self._reward_dict_list[i].keys():
            self._reward_dict_list[i][key] = []
        for key in self._info_dict_list[i].keys():
            self._info_dict_list[i][key] = []

    def _log_final_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]
        i = locals_['i']
        if infos["TimeLimit.truncated"] or infos['is_success']:
            if self.verbose > 0:
                print("INFO: Logging final metrics")

            for key in self._info_dict_list[i].keys():
                self._info_dict_list[i][key].append(infos[key])

            self._terminal_dict["init_distance"].append(self.training_env.get_attr("init_distance", i)[0])
            self._terminal_dict["init_perp_cosine_sim"].append(self.training_env.get_attr("init_perp_cosine_sim", i)[0])
            self._terminal_dict["init_point_cosine_sim"].append(self.training_env.get_attr("init_point_cosine_sim", i)[0])
            self._terminal_dict["init_angular_error"].append(self.training_env.get_attr("init_angular_error", i)[0])

            branch_loc = self.training_env.get_attr("tree_goal_pos", i)[0]
            branch_or = self.training_env.get_attr("tree_goal_or", i)[0]

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

            self.reset_buffers(i)

    def _on_step(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._log_collisions(_locals, _globals)
        self._log_rewards(_locals, _globals)
        self._log_final_metrics(_locals, _globals)

    def save_results(self):
        episode_info_df = pd.DataFrame(self._episode_info)
        terminal_info_df = pd.DataFrame(self._terminal_dict)
        save_df = pd.concat([episode_info_df, terminal_info_df], axis=1)
        save_df.to_csv("episode_info.csv", mode='a')


class GenerateResults:
    def __init__(self, model, env, set_goal_callback, log_callback, other_callbacks=None, verbose = 1):
        self.model = model
        self.env = env
        self.verbose = verbose
        self.set_goal_callback = set_goal_callback
        self.log_callback = log_callback

        self.other_callbacks = other_callbacks
        self._init_callback()

    def _init_callback(self):
        self.set_goal_callback.init_callback(self.model)
        self.log_callback.init_callback(self.model)
        if self.other_callbacks is not None:
            for callback in self.other_callbacks:
                callback.init_callback(self.model)
        self.num_episodes = len(self.set_goal_callback.dataset)

    def _main_callback(self, locals_, globals_):
        self.set_goal_callback._on_step(locals_, globals_)
        self.log_callback._on_step(locals_, globals_)
        if self.other_callbacks is not None:
            for callback in self.other_callbacks:
                callback._on_step(locals_, globals_)

    def run(self):
        if self.verbose > 0:
            print("INFO: Starting evaluation")
        import time
        start = time.time()
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=self.num_episodes,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
            callback=self._main_callback,
        )

        end = time.time()
        if self.verbose > 0:
            print(f"INFO: Evaluation took {end - start} seconds")
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

        if self.verbose > 0:
            print(f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        self.log_callback.save_results()
