#Fix this file by subclassing

import pickle

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
class CustomTrainCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, or_bins, verbose=0):
     
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





    def _init_callback(self) -> None:
        for i in range(self.training_env.num_envs):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos \
                = self._sample_tree_and_point()
            self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos)


    def _sample_tree_and_point(self):
            #Sample orientation from key in or_bins
            while True:
                orientation = random.choice(list(self.or_bins.keys()))
                if len(self.or_bins[orientation]) == 0:
                    #print("No trees in orientation", orientation)
                    continue
                tree_urdf, random_point, tree_orientation, scale = random.choice(self.or_bins[orientation])
                required_point_pos = random.choice(self.reachable_euclidean_grid)

                current_point_pos = random_point[0]
                current_branch_or = random_point[1]
                offset = np.random.uniform(-0.025, 0.025, 3)
                delta_tree_pos = np.array(required_point_pos) - np.array(current_point_pos) + offset
                final_point_pos = np.array(current_point_pos) + delta_tree_pos

                if (delta_tree_pos > self.delta_pos_max).any() or (delta_tree_pos < self.delta_pos_min).any():
                     # print("Invalid delta pos", delta_tree_pos, "required pos", required_point_pos, "current pos", current_point_pos, "offset", offset)
                     continue

                break

            return tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, delta_tree_pos

    def _update_tree_properties(self, infos):
        for i in range(len(infos)):
            if infos[i]["TimeLimit.truncated"] or infos[i]['is_success']:
                tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos\
                    = self._sample_tree_and_point()
                self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                            point_pos = final_point_pos, point_branch_or=current_branch_or,
                                            tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos)



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
            if infos[i]["TimeLimit.truncated"]:
                for key in self._info_dict.keys():
                    self._info_dict[key].append(infos[i][key])

        self._update_tree_properties(infos)
        if self._rollouts % self._train_record_freq == 0:
            #grab screen
            self._screens_buffer.append(self._grab_screen_callback(self.locals, self.globals))

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

            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos \
                = self._sample_tree_and_point()
            self.eval_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos)
        for i in range(self.record_env.num_envs):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos \
                = self._sample_tree_and_point()
            self.record_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos)

    def _sample_tree_and_point(self):
        # Sample orientation uniformly from key in or_bins
        num_or = 40
        or_list = []
        #choose every total//num_or orientations
        total_orientations = len(self.or_bins.keys())
        for i, orientation in enumerate(self.or_bins.keys()):
            if i % (total_orientations // num_or) == 0:
                or_list.append(orientation)
        while True:
            orientation = or_list[self.episode_counter % num_or]
            if len(self.or_bins[orientation]) == 0:
                print("No trees in orientation", orientation)
                self.episode_counter += 1
            tree_urdf, random_point, tree_orientation, scale = random.choice(self.or_bins[orientation])
            required_point_pos = random.choice(self.reachable_euclidean_grid)

            current_point_pos = random_point[0]
            current_branch_or = random_point[1]
            offset = np.random.uniform(-0.025, 0.025, 3)
            delta_tree_pos = np.array(required_point_pos) - np.array(current_point_pos) + offset
            final_point_pos = np.array(current_point_pos) + delta_tree_pos
            if (delta_tree_pos > self.delta_pos_max).any() or (delta_tree_pos < self.delta_pos_min).any():
                continue
            break

        return tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, delta_tree_pos

    def update_tree_properties(self, info, idx, name):
        if name == "eval" and (info["TimeLimit.truncated"] or info['is_success']):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos \
                = self._sample_tree_and_point()
            self.eval_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale,
                                         tree_pos=tree_pos)

            self.episode_counter += 1
        elif name == "record" and info["TimeLimit.truncated"] or info['is_success']:
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos \
                = self._sample_tree_and_point()
            self.record_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale,
                                         tree_pos=tree_pos)

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
        self._collisions_acceptable_buffer.append(self.eval_env.get_attr("collisions_acceptable", 0)[0])
        self._collisions_unacceptable_buffer.append(self.eval_env.get_attr("collisions_unacceptable", 0)[0])

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
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._master_callback,
            )
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
        return continue_training

