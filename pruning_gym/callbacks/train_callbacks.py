import pickle
import random

import numpy as np
import torch as th
from pruning_sb3.pruning_gym.callbacks.callbacks import PruningSetGoalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import Video


class Pruning1TreeSetGoalCallback(PruningSetGoalCallback):
    def __init__(self, tree_info,  verbose=0):
        super(Pruning1TreeSetGoalCallback, self).__init__(verbose)
        self.tree_info = tree_info

    def _init_callback(self) -> None:
        for i in range(self.training_env.num_envs):
            self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=self.tree_info[0],
                                         point_pos=self.tree_info[1], point_branch_or=self.tree_info[2],
                                         tree_orientation=self.tree_info[3], tree_scale=self.tree_info[4],
                                         tree_pos=self.tree_info[5], point_branch_normal=self.tree_info[6])

            # self.training_env.env_method("set_ur5_pose", indices=i, pos=self.tree_info[7], orien=self.tree_info[8])
    def _sample_tree_and_point(self, idx):
        return

    def _update_tree_properties(self):
        return

    def _on_step(self) -> bool:
        return


class PruningTrainSetGoalCallback(PruningSetGoalCallback):
    def __init__(self, or_bins, verbose=0):
        super(PruningTrainSetGoalCallback, self).__init__(verbose)
        self.or_bins = or_bins
        self.delta_pos_max = np.array([1, -0.675, 0])
        self.delta_pos_min = np.array([-1, -0.9525, -2])
        self.reachable_euclidean_grid = self.get_reachable_euclidean_grid(0.95, 0.05)

    def _init_callback(self) -> None:
        for i in range(self.training_env.num_envs):
            tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
                = self._sample_tree_and_point(i)
            self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                         point_pos=final_point_pos, point_branch_or=current_branch_or,
                                         tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                         point_branch_normal=current_branch_normal)

    def _sample_tree_and_point(self, idx):
        # Sample orientation from key in or_bins
        if self.verbose > 0:
            print("INFO: Sampling tree and point")
        point_sampled = False
        while not point_sampled:
            rand_vector = self.rand_direction_vector()
            orientation = self.get_bin_from_orientation(rand_vector)
            point_sampled, point = self.maybe_sample_point(orientation)

        return point

    def _update_tree_properties(self):
        infos = self.locals["infos"]
        for i in range(len(infos)):
            if infos[i]["TimeLimit.truncated"] or infos[i]['is_success']:
                if self.verbose > 1:
                    print(f"DEBUG: Updating tree in env {i} via callback")
                tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal, \
                    = self._sample_tree_and_point(i)
                self.training_env.env_method("set_tree_properties", indices=i, tree_urdf=tree_urdf,
                                             point_pos=final_point_pos, point_branch_or=current_branch_or,
                                             tree_orientation=tree_orientation, tree_scale=scale, tree_pos=tree_pos,
                                             point_branch_normal=current_branch_normal)
    def _on_step(self) -> bool:
        if self.locals['offline']:
            return
        self._update_tree_properties()  # Maybe remove infos check and pass it in an EventCallback that triggers whenever episode terminates
        return True


class PruningTrainRecordEnvCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PruningTrainRecordEnvCallback, self).__init__(verbose)
        self._screens_buffer = []

    def reset_buffer(self):
        self._screens_buffer = []

    def _grab_screen(self):
        screen = np.array(self.training_env.render()) * 255
        screen_copy = screen.reshape((screen.shape[0], screen.shape[1], 3)).astype(np.uint8)
        return screen_copy.transpose(2, 0, 1)

    def _log_screen(self):
        self._screens_buffer.append(self._grab_screen())

    def _on_step(self) -> bool:
        if self.verbose > 1:
            print("DEBUG: Recording video")
        self._log_screen()
        return True

    def _on_rollout_end(self):
        if self.verbose > 1:
            print("DEBUG: Saving video")
        self.logger.record(
            "rollout/video",
            Video(th.ByteTensor(np.array([self._screens_buffer[:-1]])), fps=10),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.reset_buffer()


class PruningCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose=0):
        super(PruningCheckpointCallback, self).__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            
            self.model.save(model_path, exclude=["_last_obs", "_last_episode_starts", "_last_original_obs"
                                                 "expert_buffer", "expert_data", "expert_batch_idx",
                                                 "rollout_buffer", "expert_batch", "dataset", "data_iter", "dataloader",])
            if self.verbose >= 1:
                print(f"Saving model checkpoint to {model_path}")
            mean_std_path = self._checkpoint_path(checkpoint_type="mean_std_", extension="pkl")
            with open(mean_std_path, "wb") as f:
                pickle.dump((self.model.policy.running_mean_var_oflow_x, self.model.policy.running_mean_var_oflow_y), f)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

        return True
