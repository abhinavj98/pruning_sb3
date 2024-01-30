# (0.13449400663375854, -0.5022612810134888, 0.5729108452796936), (0.08233322092014395, 0.08148885209843372, -0.7017698639513068, 0.7029223753490557))
# ((0.13449425995349884, -0.5022624731063843, 0.5729091167449951), (0.08233936213522093, 0.08149007539678467, -0.7017685790089195, 0.7029227970202654))
# ((0.13449451327323914, -0.5022636651992798, 0.5729073882102966), (0.08234550355528829, 0.08149129879013207, -0.7017672940054546, 0.7029232186590406))
# ((0.13449475169181824, -0.5022648572921753, 0.5729056596755981)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import *
import numpy as np
import cv2
import random
import argparse
from pruning_sb3.args.args_test import args
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch as th
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env
def get_key_pressed(env, relevant=None):
    pressed_keys = []
    events = env.pyb.con.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys


def plot(imgs, axs, **imshow_kwargs):

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
    # for i in imgs:
    #     print("Image", i[0].shape)
    num_rows = len(imgs)
    num_cols = len(imgs[0])

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            # print(img.shape, row_idx, col_idx)
            ax = axs[row_idx, col_idx]

            if img.shape[0] == 1:
                cmap = 'gray'#, vmin = 0, vmax = 255
                ax.imshow(np.asarray(img), cmap = cmap)
            else:
                ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.pause(0.001)

# Create the ArgumentParser object

# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())
parsed_args_dict = organize_args(parsed_args)
args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
if __name__ == "__main__":
    print(parsed_args_dict['args_env']['use_optical_flow'])
    print(parsed_args_dict)
    if parsed_args_dict['args_env']['use_optical_flow'] and parsed_args_dict['args_env']['optical_flow_subproc']:
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)
    add_arg_to_env('shared_var', shared_var, ['args_train', 'args_test', 'args_record'], parsed_args_dict)

    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    env = PruningEnv(**args_test, tree_count=1)

    plt.ion()
    _, axs = plt.subplots(nrows=1, ncols=5, squeeze=False)
    plt.show()
    # env.reset()
    val = np.array([0, 0, 0, 0, 0, 0])
    # Use keyboard to move the robot
    while True:
        # Read keyboard input using python input
        action = get_key_pressed(env)
        # if action is wasd, then move the robot
        if ord('a') in action:
            val = np.array([0.001, 0, 0, 0, 0, 0])
        elif ord('d') in action:
            val = np.array([-0.001, 0, 0, 0, 0, 0])
        elif ord('s') in action:
            val = np.array([0, 0.001, 0, 0, 0, 0])
        elif ord('w') in action:
            val = np.array([0, -0.001, 0, 0, 0, 0])
        elif ord('q') in action:
            val = np.array([0, 0, 0.001, 0, 0, 0])
        elif ord('e') in action:
            val = np.array([0, 0, -0.001, 0, 0, 0])
        elif ord('z') in action:
            val = np.array([0, 0, 0, 0.001, 0, 0])
        elif ord('c') in action:
            val = np.array([0, 0, 0, -0.001, 0, 0])
        elif ord('x') in action:
            val = np.array([0, 0, 0, 0, 0.001, 0])
        elif ord('v') in action:
            val = np.array([0, 0, 0, 0, -0.001, 0])
        elif ord('r') in action:
            val = np.array([0, 0, 0, 0, 0, 0.001])
        elif ord('f') in action:
            val = np.array([0, 0, 0, 0, 0, -0.001])
        elif ord('t') in action:
            env.reset()
        else:
            val = np.array([0, 0, 0, 0, 0, 0])
        # print(val)
        observation, reward, terminated, truncated, infos = env.step(val)
        grid = [th.tensor(env.prev_observation_info[
                                                    'rgb']), th.tensor(env.observation_info[
                                                    'rgb']), th.tensor(env.observation['depth_proxy'][0]*10), th.tensor(env.observation["depth_proxy"][1]*10), th.tensor(env.observation['depth_proxy'][2])]# - th.mean(th.tensor(env.observation["depth"][0], dtype=th.float32))]
        plot(grid, axs)