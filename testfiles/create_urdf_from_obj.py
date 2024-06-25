import glob
import os

type = "test"
tree = "envy"
labelled = True
if labelled:
    in_folder = os.path.join(tree, type + "_labelled_split")
    out_folder = os.path.join(tree, type + "_labelled_split")
else:
    in_folder = os.path.join(tree, type)
    out_folder = os.path.join(tree, type)
OBJ_FOLDER = os.path.join('meshes_and_urdf', 'meshes', 'trees', in_folder)
OBJ_out_FOLDER = os.path.join('meshes_and_urdf', 'urdf', 'trees', out_folder)
# #"./meshes_and_urdf/meshes/trees/" + in_folder
# OBJ_out_FOLDER = "./meshes_and_urdf/meshes/trees/" + out_folder
print(OBJ_FOLDER)
print(OBJ_out_FOLDER)
for name in glob.glob(OBJ_FOLDER + '/*.obj'):
    print(name)
    urdf_template = """<robot name="ur5e" xmlns:xacro="http://ros.org/wiki/xacro">
    <link name="tree">
        <visual>
        <geometry>
            <mesh filename="{filepath}" scale="1 1 1"/>
        </geometry>
        <material name="LightGrey">
            <color rgba="{color}"/>
        </material>
        </visual>
        <collision concave = "true">
        <geometry>
            <mesh filename="{filepath}" scale="1 1 1"/>
        </geometry>
        </collision>
        <inertial>
        <mass value="4.0"/>
        <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
        <inertia ixx="0.00443333156" ixy="0.0" ixz="0.0" iyy="0.00443333156" iyz="0.0" izz="0.0072"/>
        </inertial>
    </link>
        <link name="world"/>
    <joint name="tree_joint" type="fixed">
        <parent link="world"/>
        <child link="tree"/>
        <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/>
    </joint>
    </robot>""".format(filepath=os.path.relpath(name), color = "0.7 0.7 0.7 0.0" if labelled else "0.7 0.7 0.7 1.0")
    #make name relative rather than absolute

    file = open(os.path.join(OBJ_out_FOLDER, os.path.basename(name).split('.')[0] + ".urdf"), "w")
    print("Writing to file: ", os.path.join(OBJ_out_FOLDER, os.path.basename(name).split('.')[0] + ".urdf"))
    file.write(urdf_template)
    file.close()
