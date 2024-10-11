#Read hdf5 file and save new one with just first trajectory

import h5py
import numpy as np

expert_trajectory_path = "trajectories_test.hdf5"
new_expert_trajectory_path = "trajectories_test_1.hdf5"


new_file = h5py.File(new_expert_trajectory_path, 'w')
with h5py.File(expert_trajectory_path, 'r') as file:
    dnames = list(file.keys())
    dnames = dnames[1:]
    for name in dnames:
        expert_traj = file[name]
        new_file.create_group(name)
        for key, value in expert_traj.attrs.items():
            new_file[name].attrs[key] = value
        for key, value in expert_traj.items():
            if isinstance(value, h5py.Dataset):
                new_file[name].create_dataset(key, data=value[:])
            elif isinstance(value, h5py.Group):
                key = name + '/' + key
                new_file.create_group(key)
                for k, v in value.items():
                    new_file[key].create_dataset(k, data=v[:])
new_file.close()


# with h5py.File(new_expert_trajectory_path, 'w') as file:
#     file.create_dataset(first_traj, data=expert_traj)
