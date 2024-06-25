import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args, make_or_bins
import argparse
import pickle
import pandas as pd
from stable_baselines3.common import utils
from pruning_sb3.pruning_gym.callbacks.eval_callbacks import PruningEvalSetGoalCallback
parser = argparse.ArgumentParser()

#TODO: Not tested. Required refactoring
if __name__ == "__main__":
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, parsed_args_dict = organize_args(
        parsed_args)
    verbose = 1
    load_timestep = args_global['load_timestep']

    if args_global['load_path']:
        load_path_model = "./logs/{}/model_{}_steps.zip".format(
            args_global['load_path'], load_timestep)
        load_path_mean_std = "./logs/{}/model_mean_std_{}_steps.pkl".format(
            args_global['load_path'], load_timestep)
    else:
        load_path_model = None

    or_bins = make_or_bins(args_train, "test")

    env = make_vec_env(PruningEnv, env_kwargs=args_record, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger

    dataset = None
    if os.path.exists(f"{type}_dataset.pkl"):
        with open(f"{type}_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    eval_env = make_vec_env(PruningEnv, env_kwargs=args_record, vec_env_cls=SubprocVecEnv, n_envs=args_global["n_envs"])

    # viz_env = PruningEnv(**args_record)
    set_goal_callback = PruningEvalSetGoalCallback(or_bins=or_bins, type=type, dataset=dataset,
                                                   num_orientations=args_callback['n_eval_orientations'],
                                                   num_points_per_or=args_callback['n_points_per_orientation'],
                                                   verbose=args_callback['verbose'])
    result_df = pd.DataFrame(
        columns=["pointx", "pointy", "pointz", "or_x", "or_y", "or_z", "or_w", "is_success", "time_total",
                 "time_find_end_config", "time_find_path", ])
    # dataset = set_goal_callback.dataset

    for i in range(len(dataset) // eval_env.num_envs):
        ret = eval_env.env_method("run_rrt_connect")
        for k in range(eval_env.num_envs):
            path, tree_info, goal_orientation, timing = ret[k]
            goal_pos = tree_info[1]
            goal_or = tree_info[2]

            success = isinstance(path, list)
            if success:
                fail_mode = 1
            else:
                fail_mode = path
            result = {"pointx": goal_pos[0], "pointy": goal_pos[1], "pointz": goal_pos[2], "or_x": goal_or[0],
                      "or_y": goal_or[1], "or_z": goal_or[2], "is_success": success, "fail_mode": fail_mode}
            result.update(timing)
            result = pd.DataFrame([result])

            result_df = pd.concat([result_df, result])
        # if path:
        #     viz_env.set_tree_properties(*tree_info)
        #     viz_env.reset()
        #     viz_env.set_tree_properties(*tree_info)
        #     viz_env.step(np.array([0, 0, 0, 0, 0, 0]))
        #     print("Path found")
        #     for i in path:
        #         viz_env.ur5.set_joint_angles(i)
        #         for j in range(10):
        #             viz_env.pyb.con.stepSimulation()
        #             # time.sleep(5 / 240)
        print(i, len(dataset) // eval_env.num_envs)
        for j in range(eval_env.num_envs):
            print("Resetting", j)
            eval_env.env_method("reset", indices=j)
            print("Updating tree properties", j)
            PruningEvalSetGoalCallback.update_tree_properties(j)

    result_df.to_csv("rrt_results.csv")
#