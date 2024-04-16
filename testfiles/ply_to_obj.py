import pymeshlab
from glob import glob
import os
remove = False
folder = "envy_ply"
PLY_FOLDER = "./meshes_and_urdf/meshes/trees/"+folder
for file_path in glob(PLY_FOLDER+"/*.ply"):
    print("Processing {}".format(file_path))
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    ms.save_current_mesh(file_path[:-4]+'.obj')
    if remove:
        os.remove(file_path)