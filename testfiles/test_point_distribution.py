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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())

parsed_args_dict = organize_args(parsed_args)


def create_bins(num_latitude_bins, num_longitude_bins):
    """
    Create bins separated by 10 degrees on a unit sphere.

    Parameters:
        num_latitude_bins (int): Number of bins along the latitude direction.
        num_longitude_bins (int): Number of bins along the longitude direction.

    Returns:
        list of tuples: List of tuples where each tuple represents a bin defined by
                        (latitude_min, latitude_max, longitude_min, longitude_max).
    """
    bin_size = np.deg2rad(10)  # Convert degrees to radians

    bins = []
    for i in range(num_latitude_bins):
        lat_min = -np.pi / 2 + i * bin_size
        lat_max = -np.pi / 2 + (i + 1) * bin_size
        for j in range(num_longitude_bins):
            lon_min = -np.pi + j * bin_size
            lon_max = -np.pi + (j + 1) * bin_size
            bins.append((lat_min, lat_max, lon_min, lon_max))

    return bins


def angle_between_vectors(v1, v2):
    """
    Calculate the angle in radians between two vectors.

    Parameters:
        v1 (numpy.ndarray): First vector.
        v2 (numpy.ndarray): Second vector.

    Returns:
        float: Angle in radians between the two vectors.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def populate_bins(direction_vectors, bins):
    """
    Populate the bins based on a list of direction vectors.

    Parameters:
        direction_vectors (list of numpy.ndarray): List of direction vectors.
        bins (list of tuples): List of bins where each tuple represents a bin defined by
                               (latitude_min, latitude_max, longitude_min, longitude_max).

    Returns:
        list of lists: List of lists where each sublist represents the indices of direction vectors
                       assigned to the corresponding bin.
    """
    bin_assignments = [[] for _ in range(len(bins))]

    for i, direction_vector in enumerate(direction_vectors):
        lat_angle = np.arcsin(direction_vector[2])
        lon_angle = np.arctan2(direction_vector[1], direction_vector[0])

        # Find the closest bin based on latitude and longitude angles
        min_dist = float('inf')
        closest_bin_idx = None
        for j, bin_ in enumerate(bins):
            lat_min, lat_max, lon_min, lon_max = bin_
            if lat_min <= lat_angle <= lat_max and lon_min <= lon_angle <= lon_max:
                dist = angle_between_vectors(direction_vector,
                                             (np.sin((lat_min + lat_max) / 2), 0, np.cos((lat_min + lat_max) / 2)))
                if dist < min_dist:
                    min_dist = dist
                    closest_bin_idx = j

        # Assign the direction vector to the closest bin
        bin_assignments[closest_bin_idx].append(i)

    return bin_assignments


def visualize_sphere(bins, bin_assignments, direction_vectors):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    """
    Visualize the unit sphere with bin color representing the frequency of direction vectors.

    Parameters:
        bins (list of tuples): List of bins where each tuple represents a bin defined by
                               (latitude_min, latitude_max, longitude_min, longitude_max).
        bin_assignments (list of lists): List of lists where each sublist represents the indices of direction vectors
                                          assigned to the corresponding bin.
        direction_vectors (list of numpy.ndarray): List of direction vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the frequency of direction vectors in each bin
    frequencies = [len(bin_assignment) for bin_assignment in bin_assignments]

    # Normalize the frequencies for colormap mapping
    norm = Normalize(vmin=min(frequencies), vmax=max(frequencies))
    cmap = plt.cm.viridis
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i, bin_ in enumerate(bins):
        color = cmap(norm(frequencies[i]))  # Color based on frequency
        lat_min, lat_max, lon_min, lon_max = bin_

        # Vertices of the bin as a square patch
        vertices = [
            [np.sin(lat_min) * np.cos(lon_min), np.sin(lat_min) * np.sin(lon_min), np.cos(lat_min)],
            [np.sin(lat_min) * np.cos(lon_max), np.sin(lat_min) * np.sin(lon_max), np.cos(lat_min)],
            [np.sin(lat_max) * np.cos(lon_max), np.sin(lat_max) * np.sin(lon_max), np.cos(lat_max)],
            [np.sin(lat_max) * np.cos(lon_min), np.sin(lat_max) * np.sin(lon_min), np.cos(lat_max)]
        ]
        vertices = np.array(vertices)

        # Plot the bin
        ax.add_collection3d(Poly3DCollection([vertices], color=color, alpha=0.8))
    # Plot direction vectors
    direction_vectors = np.array(direction_vectors)
    ax.quiver(0, 0, 0, direction_vectors[:, 0]*0.8, direction_vectors[:, 1]*0.8, direction_vectors[:, 2]*0.8, color='red')
    # Create an Axes for the colorbar
    cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    fig.colorbar(sm, cax=cax, orientation='vertical')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()


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
    for i in env.trees[:1]:
        point_list.extend(i.curriculum_points[0])

    orientations = [np.array(i[1][1])/np.linalg.norm(np.array(i[1][1])) for i in point_list]
    # orientations = np.array(orientations) / np.linalg.norm(orientations, axis=1)[:, np.newaxis]

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D




    num_latitude_bins = 18  # 90 degrees / 10 degrees per bin = 9 bins
    num_longitude_bins = 36  # 360 degrees / 10 degrees per bin = 36 bins
    bins = create_bins(num_latitude_bins, num_longitude_bins)

    bin_assignments = populate_bins(orientations, bins)

    visualize_sphere(bins, bin_assignments, orientations)


    # #Visualize the orientation vectors as a 3D arrows on matplotlib
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(0, 0, 0, orientations[:, 0], orientations[:, 1], orientations[:, 2])
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    # #Label the axes
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # plt.show()




    # # convert each orientation vector to euler angles
    # angles = []
    # for orientation in orientations:
    #     # print(orientation)
    #     angles.append((np.arccos(orientation[0]), np.arccos(orientation[1]), np.arccos(orientation[2])))
    # # convert euler angles to degrees
    # angles = np.array(angles) * 180 / np.pi
    #
    # # bin the angles
    #
    # bin_size = 5
    # bins = 185 // bin_size
    # binned_angles = []
    # for angle in angles:
    #     # print(angle)
    #     binned_angles.append((int(angle[0] // bin_size), int(angle[1] // bin_size), int(angle[2] // bin_size)))
    # binned_angles = np.array(binned_angles)
    # # make grid of bins
    # grid = np.zeros((bins, bins, bins))
    # for binned_angle in binned_angles:
    #     grid[binned_angle[0], binned_angle[1], binned_angle[2]] += 1
    # # print number in each bin and bin range
    # # print(min(grid.flatten()), max(grid.flatten()))
    # # print(grid[grid>0])
    # # for i in range(bins):
    # #     for j in range(bins):
    # #         for k in range(bins):
    # #             print(f"Bin {i}, {j}, {k}: {grid[i,j,k]}")
    # # 1d histogram of binned angles individually
    # import matplotlib.pyplot as plt
    #
    # # multply x axis by bin size to get degrees
    # plt.xticks(np.arange(0, 185, 5))
    # plt.hist(binned_angles[:, 2], bins=bins)
    #
    # plt.xlabel('Binned angles in x axis')
    # plt.ylabel('Frequency')
    #
    # plt.title('Histogram of binned angles in x axis')
    # # make vertical lines at bin edges
    # for i in range(bins):
    #     plt.axvline(i, color='r')
    #
    # plt.show()


