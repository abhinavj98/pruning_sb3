# ((0.13449475169181824, -0.5022648572921753, 0.5729056596755981)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import *
import numpy as np
import argparse
from pruning_sb3.args.args_test import args
from pruning_sb3.pruning_gym.helpers import optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env

# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())

parsed_args_dict = organize_args(parsed_args)
if __name__ == "__main__":
    print(parsed_args_dict['args_env']['use_optical_flow'])
    print(parsed_args_dict)
    if parsed_args_dict['args_env']['use_optical_flow'] and parsed_args_dict['args_env']['optical_flow_subproc']:
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)
    add_arg_to_env('shared_var', shared_var, ['args_train', 'args_test', 'args_record'], parsed_args_dict)

    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    env = PruningEnv(**args_train)
    env.robot.set_joint_angles((-2.0435414506752583, -1.961562910279876, 2.1333764856444137, -2.6531903863259485,
                                -0.7777109569760938, 3.210501267258541))
    for _ in range(100):
        env.pyb_con.con.stepSimulation()

    import matplotlib.pyplot as plt

    # Write code for 2d scatter plot
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter plot of points')
    for tree in env.trees:
        for point in tree.curriculum_points[0]:
            points = np.array(point[1])
            x = points[:, 0]
            y = points[:, 2]
            ax.scatter(x, y)

    plt.show()
