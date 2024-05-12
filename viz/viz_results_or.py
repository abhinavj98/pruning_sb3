#In this file I want to read from episode_info.csv file
#Calculate latitudes and longitudes of using orientation
#Bin each latitude and for each bin calculate average perpendicular cosine sim error
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
from pruning_sb3.pruning_gym.tree import Tree

# Step 1: Read the csv file
df = pd.read_csv('episode_info.csv')

# Step 2: Extract the orientation data
# Assuming the orientation data is stored in columns 'or_x', 'or_y', 'or_z'
orientations = df[['or_x', 'or_y', 'or_z']]
# Normalize the orientation data
orientations = orientations / np.linalg.norm(orientations, axis=1)[:, np.newaxis]

# Step 3: Calculate latitudes and longitudes from orientations
# This is a placeholder. Replace this with the actual conversion logic based on your specific context.

offset = 1e-3
latitudes = np.rad2deg(np.arcsin(orientations['or_z'])) + offset
longitudes = np.rad2deg(np.arctan2(orientations['or_y'], orientations['or_x'])) + offset
# Step 4: Bin the latitude data
num_latitude_bins = 18
#18 bins from -90 to 90 degrees using numpy linspace
bins = np.linspace(-90, 90, num_latitude_bins + 1)
print(bins, latitudes)
# Assign each latitude to a bin
bin_indices = np.digitize(latitudes, bins)
print(bin_indices)


# Step 5: For each bin, calculate the average perpendicular cosine sim error
# Assuming the perpendicular cosine sim error is stored in a column 'perpendicular_cosine_sim_error'
df['bin_index'] = bin_indices
average_errors = df.groupby('bin_index')['perpendicular_cosine_sim_error'].mean()
#print number of points in each bin
print(df['bin_index'].value_counts())

#Visualize the average perpendicular cosine sim error for each bin
print(average_errors)

#Do the same for longitudes
num_longitude_bins = 36
bins = np.linspace(-180, 180, num_longitude_bins + 1)
bin_indices = np.digitize(longitudes, bins)
print(bins,longitudes)
df['bin_index'] = bin_indices
average_errors = df.groupby('bin_index')['pointing_cosine_sim_error'].mean()
print(average_errors)
print(df['bin_index'].value_counts())
# Step 6: Plot the average perpendicular cosine sim error for each bin
