#Read the contens and print the hdf5 file

import h5py
import numpy as np

filename = 'testfiles/test.hdf5'
with h5py.File(filename, 'r') as f:
    print(f.read())