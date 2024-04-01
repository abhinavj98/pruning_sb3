import glob
import os
folder = "envy/test"
OBJ_FOLDER = "./meshes_and_urdf/meshes/trees/"+folder

for name in glob.glob(OBJ_FOLDER+'/*.obj'):
    print(name)
    urdf_template = """<robot name="ur5e" xmlns:xacro="http://ros.org/wiki/xacro">
    <link name="tree">
        <visual>
        <geometry>
            <mesh filename="{filepath}" scale="1 1 1"/>
        </geometry>
        <material name="LightGrey">
            <color rgba="0.7 0.7 0.7 1.0"/>
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
    </robot>""".format(filepath = name)
    file = open("./meshes_and_urdf/urdf/trees/{}/{}.urdf".format(folder, os.path.basename(name)[:-4]), "w")
    file.write(urdf_template)
    file.close()
