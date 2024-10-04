#Read hdf5 file and save new one with just first trajectory

import h5py
import numpy as np

expert_trajectory_path = "trajectories.hdf5"
new_expert_trajectory_path = "trajectories_test.hdf5"


new_file = h5py.File(new_expert_trajectory_path, 'w')
with h5py.File(expert_trajectory_path, 'r') as file:
    dnames = list(file.keys())
    first_traj = dnames[0]
    expert_traj = file[first_traj]
    new_file.create_group(first_traj)
    for key, value in expert_traj.attrs.items():
        new_file[first_traj].attrs[key] = value
    for key, value in expert_traj.items():
        if isinstance(value, h5py.Dataset):
            new_file[first_traj].create_dataset(key, data=value[:])
        elif isinstance(value, h5py.Group):
            key = first_traj + '/' + key
            new_file.create_group(key)
            for k, v in value.items():
                new_file[key].create_dataset(k, data=v[:])
new_file.close()


# with h5py.File(new_expert_trajectory_path, 'w') as file:
#     file.create_dataset(first_traj, data=expert_traj)
