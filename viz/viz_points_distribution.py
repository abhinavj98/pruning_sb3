#In this file I want to lead orientation of points in a pkl file.
#I want to bin these orientations and plot the distribution of points in each bin.

import pickle
import glob
import numpy as np
pkl_folder = 'pkl/envy'

pkl_files = glob.glob(pkl_folder + '/*.pkl')
print(pkl_files)
orientations = []
data_list = []
for pkl_path in pkl_files:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        # self.pos = data[0]
        data_list.extend(data[2])
        # self.reachable_points = data[2]
        # print(data)
#normalize orientations
for data in data_list:
    orientations.append(data[1])
    # print(data[1])
orientations = np.array(orientations)/np.linalg.norm(orientations, axis=1)[:, np.newaxis]
#convert each orientation vector to euler angles
angles = []
for orientation in orientations:
    # print(orientation)
    angles.append((np.arccos(orientation[0]), np.arccos(orientation[1]), np.arccos(orientation[2])))
#convert euler angles to degrees
angles = np.array(angles)*180/np.pi

#bin the angles

bin_size = 5
bins = 185//bin_size
binned_angles = []
for angle in angles:
    # print(angle)
    binned_angles.append((int(angle[0]//bin_size), int(angle[1]//bin_size), int(angle[2]//bin_size)))
binned_angles = np.array(binned_angles)
#make grid of bins
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
#1d histogram of binned angles individually
import matplotlib.pyplot as plt
#multply x axis by bin size to get degrees
plt.xticks(np.arange(0, 185, 5))
plt.hist(binned_angles[:, 2], bins=bins)

plt.xlabel('Binned angles in x axis')
plt.ylabel('Frequency')

plt.title('Histogram of binned angles in x axis')
#make vertical lines at bin edges
for i in range(bins):
    plt.axvline(i, color='r')

plt.show()

