import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.baselines.baseliines_callbacks import RRTCallback
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.tree import Tree
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args
import random
import numpy as np
# PARSE ARGUMENTS
import argparse
import pickle

parser = argparse.ArgumentParser()
import pandas as pd


def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


if __name__ == "__main__":
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    parsed_args_dict = organize_args(parsed_args)
    shared_tree_list_train = []
    shared_tree_list_test = []

    load_timestep = 7296000
    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "./logs/{}/current_model_{}.zip".format(
            parsed_args_dict['args_global']['load_path'], load_timestep)
        load_path_mean_std = "./logs/{}/current_mean_std_{}.pkl".format(
            parsed_args_dict['args_global']['load_path'], load_timestep)
    else:
        load_path_model = None
    # Parse arguments from the command line
    args_global = parsed_args_dict['args_global']
    args_train = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_train'])
    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    args_record = dict(args_test, **parsed_args_dict['args_record'])
    args_callback = dict(**parsed_args_dict['args_callback'])
    print(args_callback)
    or_bins_test = None
    dataset = None
    if os.path.exists("rrt_dataset.pkl"):
        with open("rrt_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
            # shuffle the dataset
            random.shuffle(dataset)
    else:
        data_env_test = PruningEnv(**args_test, make_trees=True)
        or_bins_test = Tree.create_bins(18, 36)
        for key in or_bins_test.keys():
            for i in data_env_test.trees:
                or_bins_test[key].extend(i.or_bins[key])
        del data_env_test
        # Shuffle the data inside the bisn
        for key in or_bins_test.keys():
            random.shuffle(or_bins_test[key])
    eval_env = make_vec_env(PruningEnv, env_kwargs=args_record, vec_env_cls=SubprocVecEnv, n_envs=args_global["n_envs"])

    # viz_env = PruningEnv(**args_record)
    eval_callback = RRTCallback(eval_env, n_eval_episodes=args_callback['n_eval_episodes'], render=False,
                                or_bins=or_bins_test, dataset=dataset)
    eval_callback._init_callback()
    result_df = pd.DataFrame(
        columns=["pointx", "pointy", "pointz", "or_x", "or_y", "or_z", "or_w", "is_success", "time_total",
                 "time_find_end_config", "time_find_path", ])
    dataset = eval_callback.dataset
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
            eval_callback.update_tree_properties(j)

    result_df.to_csv("rrt_results.csv")
#
# # print(res)
# count = 0
# for _ in range(50):
#     env.reset()
#
#     res = pp.rrt_onnect(start, env.tree_goal_pos, distance_fn, sample_fn, extend_fn, collision_fn)#, radius = 0.3)#, verbose = True)
#     if res == None:
#         print("no path found")
#         continue
# # print(count)
#     else:
#         res.append(env.tree_goal_pos)
#         eval_env.con.removeBody(eval_env.sphereUid)
#         colSphereId = -1
#         eval_env.tree_goal_pos = env.tree_goal_pos
#         visualShapeId = eval_env.con.createVisualShape(eval_env.con.GEOM_SPHERE, radius=.02,rgbaColor =[1,0,0,1])
#         eval_env.sphereUid = eval_env.con.createMultiBody(0.0, colSphereId, visualShapeId, [env.tree_goal_pos[0],env.tree_goal_pos[1],env.tree_goal_pos[2]], [0,0,0,1])
#         count += 1
#         for i in res:
#             # print(i)
#             j_angles = eval_env.calculate_ik(i, goal_orientation)
#             eval_env.set_joint_angles(j_angles)
#             for j in range(10):
#                 eval_env.con.stepSimulation()
#                 time.sleep(5/240)
#             # env.render()
#             # print(eval_env.get_current_pose()[0], env.tree_point_pos, eval_env.tree_point_pos)
#         eval_env.reset()
# print(count)
