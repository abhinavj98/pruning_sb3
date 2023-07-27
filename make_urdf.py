
import os
#read all .obj files from train
dataset = "ufo"
for dataset in ["ufo", "envy"]:
    for label in ["test", "train"]:
        train_obj_path = "./ur_e_description/meshes/trees/{}/{}".format(dataset, label)
        train_obj_files = [f for f in os.listdir(train_obj_path) if os.path.isfile(os.path.join(train_obj_path, f)) and f.endswith(".obj")]
        try:
            os.mkdir("./ur_e_description/urdf/trees/{}/{}".format(dataset, label))
        except:
            print("urdf folder already exists")

        for i in train_obj_files:
            #Save urdf str as urdf file
            urdf = """<robot name="ur5e" xmlns:xacro="http://ros.org/wiki/xacro">
                <link name="tree">
                    <visual>
                    <geometry>
                        <mesh filename="./ur_e_description/meshes/trees/{0}/train/{1}.obj" scale="1 1 1"/>
                    </geometry>
                    <material name="LightGrey">
                        <color rgba="0.7 0.7 0.7 1.0"/>
                    </material>
                    </visual>
                    <collision concave = "true">
                    <geometry>
                        <mesh filename="./ur_e_description/meshes/trees/{0}/train/{1}.obj" scale="1 1 1"/>
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
            </robot>""".format(dataset, i[:-4])
            #Save file
            with open("./ur_e_description/urdf/trees/{}/{}/{}.urdf".format(dataset,label, i[:-4]), "w") as f:
                f.write(urdf)