#Read the contens and print the hdf5 file
import h5py
import glob
filename = 'C:\\Users\\abhin\\PycharmProjects\\sb3bleeding\\pruning_sb3\\expert_trajectories'
files = glob.glob(filename + "/*.hdf5")
total_trajectories = 0
for file in files:
    with h5py.File(file, 'r') as f:
        total_trajectories += len(f.keys())

print("Total trajectories: ", total_trajectories)