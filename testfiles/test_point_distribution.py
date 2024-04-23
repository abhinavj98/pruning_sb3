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
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env
import multiprocessing as mp


# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())

parsed_args_dict = organize_args(parsed_args)
if __name__ == "__main__":
    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "./logs/{}/current_model.zip".format(
            parsed_args_dict['args_global']['load_path'])
        load_path_mean_std = "./logs/{}/current_mean_std.pkl".format(
            parsed_args_dict['args_global']['load_path'])
    else:
        load_path_model = None

    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_train'])
    env = PruningEnv(**args_test, make_trees=True)
    #get all trees and points
    point_list = []
    for i in env.trees:
        point_list.extend(i.curriculum_points[0])

    orientations = [i[1][1] for i in point_list]
    orientations = np.array(orientations) / np.linalg.norm(orientations, axis=1)[:, np.newaxis]
    # convert each orientation vector to euler angles
    angles = []
    for orientation in orientations:
        # print(orientation)
        angles.append((np.arccos(orientation[0]), np.arccos(orientation[1]), np.arccos(orientation[2])))
    # convert euler angles to degrees
    angles = np.array(angles) * 180 / np.pi

    # bin the angles

    bin_size = 5
    bins = 185 // bin_size
    binned_angles = []
    for angle in angles:
        # print(angle)
        binned_angles.append((int(angle[0] // bin_size), int(angle[1] // bin_size), int(angle[2] // bin_size)))
    binned_angles = np.array(binned_angles)
    # make grid of bins
    grid = np.zeros((bins, bins, bins))
    for binned_angle in binned_angles:
        grid[binned_angle[0], binned_angle[1], binned_angle[2]] += 1
    # print number in each bin and bin range
    # print(min(grid.flatten()), max(grid.flatten()))
    # print(grid[grid>0])
    # for i in range(bins):
    #     for j in range(bins):
    #         for k in range(bins):
    #             print(f"Bin {i}, {j}, {k}: {grid[i,j,k]}")
    # 1d histogram of binned angles individually
    import matplotlib.pyplot as plt

    # multply x axis by bin size to get degrees
    plt.xticks(np.arange(0, 185, 5))
    plt.hist(binned_angles[:, 2], bins=bins)

    plt.xlabel('Binned angles in x axis')
    plt.ylabel('Frequency')

    plt.title('Histogram of binned angles in x axis')
    # make vertical lines at bin edges
    for i in range(bins):
        plt.axvline(i, color='r')

    plt.show()


