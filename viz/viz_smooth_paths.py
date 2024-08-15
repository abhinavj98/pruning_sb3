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
from pruning_sb3.pruning_gym.helpers import set_args, organize_args, convert_string
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
    #create an empty df with columns 'x', 'y', 'z'
    new_df = pd.DataFrame(columns=['x', 'y', 'z'])
    new_df['x'] = df['orientation'].apply(lambda x: x[0])
    new_df['y'] = df['orientation'].apply(lambda x: x[1])
    new_df['z'] = df['orientation'].apply(lambda x: x[2])
    orientations = new_df

    # Normalize the orientation data
    orientations = orientations / np.linalg.norm(orientations, axis=1)[:, None]
    print(orientations)
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


    return bins

def populate_euclidean_grid(grid, df, title):
    for i, pos in enumerate(df['pos'].values):
        # Find the closest point in the grid
        closest_center = min(grid.keys(), key=lambda x: np.linalg.norm(np.array(x) - pos))
        grid[closest_center].append(df['trajectory_in_frame'][i])
    return grid


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


def visualize_2d(bins, idx_name, title):
    # Extract latitude and longitude ranges from each bin

    lat_centers, lon_centers = zip(*bins.keys())

    # Flatten bin assignments to get frequencies
    frequencies = [bin_assignment for bin_assignment in bins.values()]

    # Convert to numpy arrays for hist2d
    lat_centers = np.deg2rad(np.array(lat_centers))
    lon_centers = np.deg2rad(np.array(lon_centers))
    frequencies = np.array(frequencies)

    # Create 2D histogram
    figure = plt.figure()
    plt.hist2d(lat_centers, lon_centers, weights=frequencies, bins=[num_latitude_bins, num_longitude_bins])
    plt.colorbar(label=idx_name)
    plt.xlabel('Latitude (rad)')
    plt.ylabel('Longitude (rad)')
    # import seaborn as sns
    # sns.histplot(x=lat_centers, y=lon_centers, weights=frequencies, binwidth=np.deg2rad(11), cbar=True, palette=pallet)

    # plt.title(idx_name)
    # plt.savefig('grid{}.png'.format(idx_name), bbox_inches='tight', pad_inches=0.05)
    plt.show()

def visualize_euclidean_bins(grid, idx, title):
    #Visualize the euclidean grid in x,y bins
    x, z = zip(*grid.keys())
    values = [bin_assignment for bin_assignment in grid.values()]
    values = np.array(values)
    figure = plt.figure()
    plt.hist2d(np.array(x), np.array(z), weights=values, bins=[36, 18], range = [[-0.9, 0.9], [0.9, 1.8]])
    plt.colorbar(label = idx)
    plt.xlabel('Left')
    plt.ylabel('Up')
    plt.title(title)
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


def plot_bar(df, label, save=False):
    # is_success rate as a bar plot using seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    palette = [(0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.466, 0.674, 0.188)]
    # sns.set_palette(palette)

    sns.barplot(data=df, y='Success Rate', hue='Environment', palette=palette)
    # Remove title from legend
    plt.legend(title=None)
    plt.ylabel('Success Rate')
    # plt.title(label)
    if save:
        plt.savefig('bar{}.png'.format(label), bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()


def get_reachable_euclidean_grid(radius, resolution):
    num_bins = round(radius / resolution) * 2
    base_center = np.array([0, 0, 0.91])
    # Create a 3D grid of indices
    i, j, k = np.mgrid[-num_bins // 2:num_bins // 2, -num_bins // 2:num_bins // 2, -num_bins // 2:num_bins // 2]

    # Scale and shift the grid to get the centers of the bins
    centers = np.stack([(i + 0.5) * resolution, (j + 0.5) * resolution, (k + 0.5) * resolution],
                       axis=-1)
    # centers = centers.reshape(-1, 3)
    # Create a mask for the valid centers
    mask = (centers[..., 0] ** 2 + centers[..., 1] ** 2 + centers[..., 2] ** 2 <= radius ** 2) & (
            centers[..., 1] < -0.7) & (centers[..., 2] > -0.05)

    # Apply the mask to the centers array to get the valid centers
    valid_centers = centers[mask] + base_center
    #Make a dictionary of the valid centers
    grid = {tuple(center): [] for center in valid_centers}

    return grid

# def get_reachable_euclidean_grid(radius, resolution):
#     num_bins = round(radius / resolution) * 2
#     base_center = np.array([0, 0.91])
#     # Create a 3D grid of indices
#     i, j = np.mgrid[-num_bins // 2:num_bins // 2, -num_bins // 2:num_bins // 2]
#
#     # Scale and shift the grid to get the centers of the bins
#     centers = np.stack([(i + 0.5) * resolution, (j + 0.5) * resolution],
#                        axis=-1)
#     centers = centers.reshape(-1, 3)
#     # Create a mask for the valid centers
#     # mask = (centers[..., 0] ** 2 + centers[..., 1] ** 2 + centers[..., 2] ** 2 <= radius ** 2) & (
#     #         centers[..., 1] < -0.7) & (centers[..., 2] > -0.05)
#
#     # Apply the mask to the centers array to get the valid centers
#     valid_centers = centers + base_center
#     #Make a dictionary of the valid centers
#     grid = {tuple(center): [] for center in valid_centers}
#
#     return grid


df_smooth = pd.read_csv('rrt_connect_paths_goal_smoothed.csv')

df_smooth['tree_info'] = df_smooth['tree_info'].apply(convert_string)
df_smooth['path'] = df_smooth['path'].apply(convert_string)

#tree_urdf, final_point_pos, current_branch_or, tree_orientation, scale, tree_pos, current_branch_normal = tree_info
#Make a new column for each of the tree_info
df_smooth['tree_urdf'] = df_smooth['tree_info'].apply(lambda x: x[0])
df_smooth['pos'] = df_smooth['tree_info'].apply(lambda x: x[1])
df_smooth['orientation'] = df_smooth['tree_info'].apply(lambda x: x[2])
df_smooth['tree_orientation'] = df_smooth['tree_info'].apply(lambda x: x[3])
df_smooth['scale'] = df_smooth['tree_info'].apply(lambda x: x[4])
df_smooth['tree_pos'] = df_smooth['tree_info'].apply(lambda x: x[5])
df_smooth['current_branch_normal'] = df_smooth['tree_info'].apply(lambda x: x[6])



euclidean_grid = get_reachable_euclidean_grid(0.9, 0.05)
euclidean_grid = populate_euclidean_grid(euclidean_grid, df_smooth, 'asd')
# x, y, z = zip(*grid.keys())
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #Label the axes
# ax.set_xlabel('Left')
# ax.set_ylabel('Forward')
# ax.set_zlabel('Up')
#
# ax.scatter(x, y, z)
# plt.show()

#scatter plot pos
# x, y, z = zip(*df_smooth['pos'])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Left')
# ax.set_ylabel('Forward')
# ax.set_zlabel('Up')
#
# ax.scatter(x, y, z)
# plt.show()
#
# x = df_smooth['pos'].apply(lambda x: x[0])
# z = df_smooth['pos'].apply(lambda x: x[2])
# plt.scatter(x, z)
# plt.show()
mean_bins = {}
for key in euclidean_grid.keys():
    new_key = (key[0], key[2])
    if len(euclidean_grid[key]) > 0:
        mean_bins[new_key] = np.max(euclidean_grid[key])
    else:
        mean_bins[new_key] = 0

    # print(key, mean_bins[?key])
visualize_euclidean_bins(mean_bins, '% trajectory with goal in frame', '% trajectory with goal in frame')


num_latitude_bins = 18
num_longitude_bins = 36
bins = create_bins(num_latitude_bins, num_longitude_bins)
bins = populate_bins(bins, df_smooth, 'trajectory_in_frame')
perp_bins = {}
for key in bins.keys():
    # print(key, bins[key])
    if len(bins[key]) > 0:
        perp_bins[key] = np.max(bins[key])
    else:
        perp_bins[key] = 0
visualize_2d(perp_bins, '% trajectory with goal in frame', '% trajectory with goal in frame')
print(df_smooth.columns)
print(np.mean(df_smooth['len_refined_path']))
print(np.mean(df_smooth['trajectory_in_frame']))
print(len(df_smooth))
print(df_smooth['trajectory_in_frame'])
print(len(df_smooth[df_smooth['trajectory_in_frame'] > 0.3]))