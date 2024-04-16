
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
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env
import multiprocessing as mp
from pruning_sb3.pruning_gym.optical_flow import OpticalFlow
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            try:
                img = F.to_pil_image(img.to("cpu"))
            except:
                img = F.to_pil_image(img)
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #Label the axes
    # axs[0, 0].set_ylabel("Input")
    # axs[0, 1].set_ylabel("Predicted Flow")
    # axs[0, 2].set_ylabel("Predicted Flow CV")

    plt.tight_layout()
    plt.show()

def get_key_pressed(env, relevant=None):
    pressed_keys = []
    events = env.pyb.con.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys


# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())

parsed_args_dict = organize_args(parsed_args)
if __name__ == "__main__":
    manager = mp.Manager()
    shared_list = manager.list()
    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "./logs/{}/current_model.zip".format(
            parsed_args_dict['args_global']['load_path'])
        load_path_mean_std = "./logs/{}/current_mean_std.pkl".format(
            parsed_args_dict['args_global']['load_path'])
    else:
        load_path_model = None
    add_arg_to_env('shared_tree_list', shared_list, ['args_train', 'args_test', 'args_record'], parsed_args_dict)
    of = OpticalFlow()
    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    env = PruningEnv(**args_test, make_trees=True)
    env.action_scale = 1
    env.ur5.set_joint_angles((-2.0435414506752583, -1.961562910279876, 2.1333764856444137, -2.6531903863259485,
                              -0.7777109569760938, 3.210501267258541))
    for _ in range(100):
        env.pyb.con.stepSimulation()
    # env.reset()
    val = np.array([0, 0, 0, 0, 0, 0])
    # Use keyboard to move the robot
    while True:
        # Read keyboard input using python input
        action = get_key_pressed(env)
        # if action is wasd, then move the robot
        if ord('a') in action:
            val = np.array([0.05, 0, 0, 0, 0, 0])
        elif ord('d') in action:
            val = np.array([-0.05, 0, 0, 0, 0, 0])
        elif ord('s') in action:
            val = np.array([0, 0.05, 0, 0, 0, 0])
        elif ord('w') in action:
            val = np.array([0, -0.05, 0, 0, 0, 0])
        elif ord('q') in action:
            val = np.array([0, 0, 0.05, 0, 0, 0])
        elif ord('e') in action:
            val = np.array([0, 0, -0.05, 0, 0, 0])
        elif ord('z') in action:
            val = np.array([0, 0, 0, 0.05, 0, 0])
        elif ord('c') in action:
            val = np.array([0, 0, 0, -0.05, 0, 0])
        elif ord('x') in action:
            val = np.array([0, 0, 0, 0, 0.05, 0])
        elif ord('v') in action:
            val = np.array([0, 0, 0, 0, -0.05, 0])
        elif ord('r') in action:
            val = np.array([0, 0, 0, 0, 0, 0.05])
        elif ord('f') in action:
            val = np.array([0, 0, 0, 0, 0, -0.05])
        elif ord('t') in action:
            env.reset()
        else:
            val = np.array([0.05, 0.0, 0., 0., 0., 0.])
        # print(val)
        import torch
        observation, reward, terminated, truncated, infos = env.step(val)
        img = torch.tensor(np.moveaxis(observation['rgb'], -1,0)).unsqueeze(0)
        img_prev = torch.tensor(np.moveaxis(observation['prev_rgb'], -1,0)).unsqueeze(0)
        print(img.shape, img_prev.shape)
        flow = of.calculate_optical_flow(img, img_prev)
        #display the flow
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        print(flow[0][0].min(), flow[0][0].max(), flow[0][1].min(), flow[0][1].max())
        plot([[img[0], img_prev[0], flow[0][0], flow[0][1]]])