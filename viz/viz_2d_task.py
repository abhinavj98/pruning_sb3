# In this file I want to read from episode_info.csv file
# Calculate latitudes and longitudes of using orientation
# Bin each latitude and for each bin calculate average perpendicular cosine sim error
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
from pruning_sb3.pruning_gym.models import *
import numpy as np
import argparse
from pruning_sb3.args.args import args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args
import matplotlib.pyplot as plt

# %%
# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
args, unknown = parser.parse_known_args()
parsed_args = vars(args)
print(args)
parsed_args_dict = organize_args(parsed_args)

# %%
import math


def get_bin_from_orientation(orientation):
    offset = 1e-4
    orientation = orientation / np.linalg.norm(orientation)
    lat_angle = np.rad2deg(np.arcsin(orientation[2])) + offset
    lon_angle = np.rad2deg(np.arctan2(orientation[1], orientation[0])) + offset
    lat_angle_min = rounddown(lat_angle)
    lat_angle_max = roundup(lat_angle)
    lon_angle_min = rounddown(lon_angle)
    lon_angle_max = roundup(lon_angle)
    bin_key = (round((lat_angle_min + lat_angle_max) / 2), round((lon_angle_min + lon_angle_max) / 2))
    # if bin_key[0] not in between -85 and 85 set as 85 or -85
    # if bin_keyp[1] not in between -175 and 175 set as 175 or -175

    if bin_key[0] > 85:
        bin_key = (85, bin_key[1])
    elif bin_key[0] < -85:
        bin_key = (-85, bin_key[1])
    if bin_key[1] > 175:
        bin_key = (bin_key[0], 175)
    elif bin_key[1] < -175:
        bin_key = (bin_key[0], -175)
    return bin_key


def roundup(x):
    return math.ceil(x / 10.0) * 10


def rounddown(x):
    return math.floor(x / 10.0) * 10


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
    offset = np.deg2rad(1)
    bins = {}
    for i in range(num_latitude_bins):
        lat_min = np.rad2deg(-np.pi / 2 + i * bin_size)
        lat_max = np.rad2deg(-np.pi / 2 + (i + 1) * bin_size)
        for j in range(num_longitude_bins):
            lon_min = np.rad2deg(-np.pi + j * bin_size)
            lon_max = np.rad2deg(-np.pi + (j + 1) * bin_size)
            bins[(round((lat_min + lat_max) / 2), round((lon_min + lon_max) / 2))] = []

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


def populate_bins(bins, df, idx_name):
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
    orientations = df[['or_x', 'or_y', 'or_z']]
    # Normalize the orientation data
    orientations = orientations / np.linalg.norm(orientations, axis=1)[:, np.newaxis]

    offset = 1e-3
    for i, direction_vector in enumerate(orientations.values):
        lat_angle = np.rad2deg(np.arcsin(direction_vector[2])) + offset
        lon_angle = np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])) + offset
        lat_angle_min = rounddown(lat_angle)
        lat_angle_max = roundup(lat_angle)
        lon_angle_min = rounddown(lon_angle)
        lon_angle_max = roundup(lon_angle)
        bin_key = (round((lat_angle_min + lat_angle_max) / 2), round((lon_angle_min + lon_angle_max) / 2))
        if bin_key[0] > 85:
            bin_key = (85, bin_key[1])
        elif bin_key[0] < -85:
            bin_key = (-85, bin_key[1])
        if bin_key[1] > 175:
            bin_key = (bin_key[0], 175)
        elif bin_key[1] < -175:
            bin_key = (bin_key[0], -175)
        # append perp_cosine_sim_error to the bin
        bins[bin_key].append((df[idx_name][i]))

        # Find the closest bin based on latitude and longitude angles
        #
        # min_dist = float('inf')
        # closest_bin_idx = None
        # for j, bin_ in enumerate(bins):
        #     lat_min, lat_max, lon_min, lon_max = bin_
        #     if lat_min <= lat_angle <= lat_max and lon_min <= lon_angle <= lon_max:
        #         dist = angle_between_vectors(direction_vector,
        #                                      (np.sin((lat_min + lat_max) / 2), 0, np.cos((lat_min + lat_max) / 2)))
        #         if dist < min_dist:
        #             min_dist = dist
        #             closest_bin_idx = j
        #
        # # Assign the direction vector to the closest bin
        # bin_assignments[closest_bin_idx].append(i)

    return bins


def visualize_sphere(bins, idx_name):
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

    # Compute colors
    frequencies = [bin_assignment for bin_assignment in bins.values()]
    norm = plt.Normalize(vmin=min(frequencies), vmax=max(frequencies))
    cmap = plt.cm.viridis
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for i, bin_ in enumerate(bins.keys()):
        lat_center, lon_center = bin_
        lat_min = np.deg2rad(lat_center - 5)
        lat_max = np.deg2rad(lat_center + 5)
        lon_min = np.deg2rad(lon_center - 5)
        lon_max = np.deg2rad(lon_center + 5)
        color = scalar_map.to_rgba(frequencies[i])

        # Draw rectangle on the sphere
        u = np.linspace(lon_min, lon_max, 10)
        v = np.linspace(lat_min, lat_max, 10)
        x = np.outer(np.cos(u), np.cos(v))
        y = np.outer(np.sin(u), np.cos(v))
        z = np.outer(np.ones(np.size(u)), np.sin(v))
        ax.plot_surface(x, y, z, color=color, alpha=1)

    # Plot direction vectors
    # direction_vectors = np.array(direction_vectors)
    # ax.quiver(0, 0, 0, direction_vectors[:, 0], direction_vectors[:, 1], direction_vectors[:, 2], color='red')

    ax.set_xlabel('X - Branch direction')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z - Up')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    fig.colorbar(scalar_map, cax=cax, orientation='vertical')
    scalar_map.set_array(frequencies)
    plt.show()


def visualize_2d(bins, idx_name, title, colorbar_title, reverse=False, save=False):
    # Extract latitude and longitude ranges from each bin

    lat_centers, lon_centers = zip(*bins.keys())

    # Flatten bin assignments to get frequencies
    frequencies = [bin_assignment for bin_assignment in bins.values()]

    # Convert to numpy arrays for hist2d
    lat_centers = np.deg2rad(np.array(lat_centers))
    lon_centers = np.deg2rad(np.array(lon_centers))
    frequencies = np.array(frequencies)

    # Create 2D histogram
    # figure = plt.figure()
    # invert colormap

    pallet = plt.cm.get_cmap('viridis', 256)
    if reverse:
        pallet = pallet.reversed()

    plt.hist2d(lat_centers, lon_centers, weights=frequencies, bins=[num_latitude_bins, num_longitude_bins], cmap=pallet)
    plt.colorbar(label=colorbar_title)
    plt.xlabel('Elevation ({}) (rad)'.format('\u03B8'))
    plt.ylabel('Azimuth ({}) (rad)'.format('\u03C6'))
    plt.title(title)
    # Set legend title to idx_name
    # import seaborn as sns
    # sns.histplot(x=lat_centers, y=lon_centers, weights=frequencies, binwidth=np.deg2rad(11), cbar=True, palette=pallet)

    # plt.title(idx_name)
    if save:
        plt.savefig('grid{}.png'.format(title), bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    # convert M to euler angles
    return M @ [0, 0, 1]

    return M


def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def plot_bar(df, label):
    # is_success rate as a bar plot using seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    palette = [(0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.466, 0.674, 0.188)]
    # sns.set_palette(palette)

    sns.barplot(data=df, y='Success Rate', hue='Environment', palette=palette)
    plt.ylabel('Success Rate')
    plt.title(label)
    # plt.savefig('bar{}.png'.format(label), bbox_inches='tight', pad_inches=0.05)
    # Different colors for the bars

    plt.show()


# Step 1: Read the csv file
df_policy = pd.read_csv('results_data/policy_task.csv')
df_rrt = pd.read_csv('results_data/rrt_task.csv')
df_policy['pointing_cosine_angle_error_abs'] = np.arccos(df_policy['pointing_cosine_sim_error']).abs()
df_policy['perpendicular_cosine_angle_error_abs'] = np.arccos(df_policy['perpendicular_cosine_sim_error']).abs()
# print(df_policy.keys())
# print(df_rrt.head())

# Make a common table for both dfs where 'pointx', 'pointy', 'pointz', 'or_x', 'or_y', 'or_z','or_w' are same an
new_df = pd.merge(df_policy, df_rrt, on=['pointx', 'pointy', 'pointz', 'or_x', 'or_y', 'or_z'])
new_df['is_success'] = new_df['is_success_x'] | new_df['is_success_y']

new_df = new_df[new_df['is_success'] == True]

filtered_df_policy = pd.merge(df_policy, new_df[['pointx', 'pointy', 'pointz', 'or_x', 'or_y', 'or_z']],
                              on=['pointx', 'pointy', 'pointz', 'or_x', 'or_y', 'or_z'],
                              how='inner')
filtered_df_rrt = pd.merge(df_rrt, new_df[['pointx', 'pointy', 'pointz', 'or_x', 'or_y', 'or_z']],
                           on=['pointx', 'pointy', 'pointz', 'or_x', 'or_y', 'or_z'],
                           how='inner')
print(len(filtered_df_policy[filtered_df_policy['is_success'] == True]) / len(filtered_df_policy))
print(len(filtered_df_rrt[filtered_df_rrt['is_success'] == True]) / len(filtered_df_rrt))
# Normalize the orientation data
# orientations = orientations / np.linalg.norm(orientations, axis=1)[:, np.newaxis]
dataset = []

num_latitude_bins = 18
num_longitude_bins = 36

idx_list = ['pointing_cosine_angle_error_abs', 'perpendicular_cosine_angle_error_abs', 'is_success']
title_list = ['Pointing Angle Error (rad)', 'Perpendicular Angle Error (rad)', 'Success Rate (Policy)']
colorbar_title_list = ['Pointing Angle Error (rad)', 'Perpendicular Angle Error (rad)', 'Success Rate']
idx_title_list = zip(idx_list, title_list, colorbar_title_list)

for idx_name, idx_title, colorbar_title in idx_title_list:
    # For each bin, calculate the average perpendicular cosine sim error and display it
    plt.figure()
    bins = create_bins(num_latitude_bins, num_longitude_bins)
    bins = populate_bins(bins, df_policy, idx_name)

    # For each bin, calculate the average perpendicular cosine sim error and display it
    perp_bins = {}
    for key in bins.keys():
        perp_bins[key] = np.mean(np.array(bins[key]))

    title = idx_title
    if 'success' in idx_name:
        reverse = False
    else:
        reverse = True
    visualize_2d(perp_bins, idx_name, title, colorbar_title=colorbar_title, reverse=reverse, save=True)

# bins = create_bins(num_latitude_bins, num_longitude_bins)
# bins = populate_bins(bins, filtered_df_policy, idx_name)

# for key in bins.keys():
#     if len(bins[key]) == 0:
#         bins[key].append(1)
#     perp_bins[key] = np.mean(np.array(bins[key]))
# visualize_2d(perp_bins, 'is_success', 'Success Rate', reverse=False, save=False)
# visualize_2d is_success rate for RRT
bins = create_bins(num_latitude_bins, num_longitude_bins)
bins = populate_bins(bins, df_rrt, 'is_success')
perp_bins = {}
title = 'Success Rate (RRT Connect)'
colorbar_title = 'Success Rate'
for key in bins.keys():
    perp_bins[key] = np.mean(np.array(bins[key]))
plt.figure()
visualize_2d(perp_bins, 'is_success', title, colorbar_title, save=True)
#
# bins_rrt = create_bins(num_latitude_bins, num_longitude_bins)
# bins_rrt = populate_bins(bins_rrt, df_rrt, 'is_success')
# # perp_bins_rrt = {}
# # for key in bins_rrt.keys():
# #     perp_bins_rrt[key] = np.mean(np.array(bins_rrt[key]))
#
# bins_policy = create_bins(num_latitude_bins, num_longitude_bins)
# bins_policy = populate_bins(bins_policy, df_policy, 'is_success')
# perp_bins_policy = {}
# for key in bins_policy.keys():
#     perp_bins_policy[key] = np.mean(np.array(bins_policy[key]))

# reachable_bins = {}
# for key in bins_rrt.keys():
#     for i in range(len(bins_rrt[key])):
#         print(len(bins_rrt[key]), len(bins_policy[key]))
#         if key not in reachable_bins:
#             reachable_bins[key] = []
#         reachable_bins[key].append((bins_rrt[key][i] or bins_policy[key][i]))
# #threshold at 0.5
# threshold = 0.5
# perp_bins = {}
# for key in reachable_bins.keys():
#     perp_bins[key] = np.mean(np.array(reachable_bins[key]))
# visualize_2d(reachable_bins, 'is_success', 'Reachable Success Rate')
# for key in bins_rrt.keys():
#     if perp_bins_rrt[key] >= threshold:
#         perp_bins_rrt[key] = 1
#     else:
#         perp_bins_rrt[key] = 0
# for key in bins_policy.keys():
#     if perp_bins_policy[key] >= threshold:
#         perp_bins_policy[key] = 1
#     else:
#         perp_bins_policy[key] = 0
#
# visualize_2d(perp_bins_rrt, 'is_success', 'Success Rate RRT')
# visualize_2d(perp_bins_policy, 'is_success', 'Success Rate Policy')
