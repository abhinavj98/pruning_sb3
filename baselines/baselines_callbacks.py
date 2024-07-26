import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import pandas as pd
from pruning_sb3.pruning_gym.callbacks.eval_callbacks import PruningEvalSetGoalCallback
class PruningRRTSetGoalCallback(PruningEvalSetGoalCallback):
    def __init__(self, or_bins, type, num_orientations, num_points_per_or, dataset, verbose=0):
        super(PruningRRTSetGoalCallback, self).__init__(or_bins=or_bins, type=type, dataset = dataset, num_orientations=num_orientations,
                                                        num_points_per_or=num_points_per_or, verbose=verbose)

    def update_tree_properties(self, idx):
        tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal \
            = self._sample_tree_and_point(idx)

        self.training_env.env_method("set_tree_properties", indices=idx, tree_urdf=tree_urdf,
                                     point_pos=final_point_pos, point_branch_or=current_branch_or,
                                     tree_orientation=tree_orientation, tree_scale=scale,
                                     tree_pos=tree_pos, point_branch_normal=current_branch_normal)


    def init_callback(self, env):
        self.training_env = env
        self._init_callback()

    @property
    def training_env(self):
        return self._training_env

    @training_env.setter
    def training_env(self, value):
        self._training_env = value
class GenerateResults:
    def __init__(self, env, set_goal_callback, planner, shortcutting = False, save_video=False):
        self.set_goal_callback = set_goal_callback
        self.set_goal_callback.init_callback(env)
        self.save_video = save_video
        self.planner = planner
        self.shortcutting = shortcutting
        # self.result_df = pd.DataFrame(
        #     columns=["pointx", "pointy", "pointz", "or_x", "or_y", "or_z", "or_w", "is_success", "time_total",
        #              "time_find_end_config", "time_find_path", ])

    def run(self, file_path):
        num_points = len(self.set_goal_callback.dataset)//self.set_goal_callback.training_env.num_envs
        print("Running", num_points, "points")
        ret = self.set_goal_callback.training_env.env_method("run_baseline", planner = self.planner, file_path = file_path, save_video = self.save_video, shortcutting = self.shortcutting)
        # for i in ret:
        #     self.result_df = pd.concat([self.result_df, i])
        #
        # self.result_df.to_csv("rrt_results.csv")
