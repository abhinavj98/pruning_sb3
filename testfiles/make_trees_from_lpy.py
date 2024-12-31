#Step 1: Convert to obj from ply
import os
from glob import glob
import pymeshlab
import shutil
#Read command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--step", help="Which step to run", type=int, default=1)
args = parser.parse_args()
step = args.step

type = "ufo"
input_folder = os.path.join("meshes_and_urdf", "meshes", "trees", type)
ply_folder = os.path.join(input_folder, "ply")
obj_folder = os.path.join(input_folder, "obj")
os.makedirs(obj_folder, exist_ok=True)

total_files = len(glob(os.path.join(obj_folder, "*.obj")))
train_folder = os.path.join(input_folder, "train")
test_folder = os.path.join(input_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
test_labelled_folder = os.path.join(input_folder, "test_labelled")
train_labelled_folder = os.path.join(input_folder, "train_labelled")
os.makedirs(test_labelled_folder, exist_ok=True)
os.makedirs(train_labelled_folder, exist_ok=True)
test_labelled_split_folder = os.path.join(input_folder, "test_labelled_split")
train_labelled_split_folder = os.path.join(input_folder, "train_labelled_split")
os.makedirs(test_labelled_split_folder, exist_ok=True)
os.makedirs(train_labelled_split_folder, exist_ok=True)
urdf_folder = os.path.join("meshes_and_urdf", "urdf", "trees", type)
urdf_folder_train = os.path.join(urdf_folder, "train")
urdf_folder_test = os.path.join(urdf_folder, "test")
os.makedirs(urdf_folder_train, exist_ok=True)
os.makedirs(urdf_folder_test, exist_ok=True)
urdf_folder_train_labelled_split = os.path.join(urdf_folder, "train_labelled_split")
urdf_folder_test_labelled_split = os.path.join(urdf_folder, "test_labelled_split")
os.makedirs(urdf_folder_train_labelled_split, exist_ok=True)
os.makedirs(urdf_folder_test_labelled_split, exist_ok=True)



import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pywavefront
from collections import defaultdict
from pruning_sb3.pruning_gym import label

def parse_labelled_tree(file_path):
    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    vertices = scene.vertices
    vertices = [vertex[:3] for vertex in vertices]
    faces = scene.mesh_list[0].faces
    vertex_colors = []

    for line in open(file_path, 'r'):
        if line.startswith('v '):
            color = line.split()[4:]
            vertex_colors.append(tuple(map(float, color)))

    return vertices, faces, vertex_colors, scene


def group_by_color(vertices, faces, vertex_colors):
    """Group vertices and faces by color."""
    color_groups = defaultdict(lambda: {'vertices': [], 'faces': []})

    for i, (vertex, color) in enumerate(zip(vertices, vertex_colors)):
        color_groups[color]['vertices'].append((i, vertex))

    for face in faces:
        face_colors = set(vertex_colors[vi] for vi in face)
        if len(face_colors) == 1:
            color = face_colors.pop()
            color_groups[color]['faces'].append(face)

    return color_groups


def create_mesh_data(color_groups):
    """Create mesh data for each color group."""
    mesh_data = {}

    for color, group in color_groups.items():
        vertices = []
        indices = []
        vertex_map = {}

        for i, (index, vertex) in enumerate(group['vertices']):
            vertex_map[index] = i
            vertices.append(vertex)

        for face in group['faces']:
            indices.append([vertex_map[vi] for vi in face])

        mesh_data[color] = (vertices, indices)
        # if self.verbose > 1:
        print(f"Color: {color}, Vertices: {len(vertices)}, Faces: {len(indices)}")

    return mesh_data


def split_obj_by_color(input_file):
    vertices, faces, vertex_colors, _ = parse_labelled_tree(input_file)
    color_groups = group_by_color(vertices, faces, vertex_colors)
    mesh_data = create_mesh_data(color_groups)
    return mesh_data


def save_as_obj(vertices, indices, output_folder, output_file, label, color):
    print(output_folder, output_file, label)
    save_file = os.path.join(output_folder, output_file+'_'+label + '.obj')
    print(f"Saving to {save_file}")
    #Add vertex colors too
    with open(save_file, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")

        for face in indices:
            f.write(f"f {' '.join(str(i + 1) for i in face)}\n")

def generate_urdf(filepath, labelled):
    """
    Generate a URDF string for the given OBJ file.

    Args:
        filepath (str): The relative path to the OBJ file.
        labelled (bool): If True, set material transparency to 0.0.

    Returns:
        str: The URDF content as a string.
    """
    color = "0.7 0.7 0.7 0.0" if labelled else "0.7 0.7 0.7 1.0"
    return f"""<robot name="ur5e" xmlns:xacro="http://ros.org/wiki/xacro">
    <link name="tree">
        <visual>
            <geometry>
                <mesh filename="{filepath}" scale="1 1 1"/>
            </geometry>
            <material name="LightGrey">
                <color rgba="{color}"/>
            </material>
        </visual>
        <collision concave="true">
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
    </robot>"""


if step == 1:
    for file_path in glob(os.path.join(ply_folder, "*.ply")):
        print("Processing {}".format(file_path))

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
        output_file = os.path.join(obj_folder, os.path.basename(file_path)[:-4] + '.obj')
        print(f"Saving to {output_file}")
        ms.save_current_mesh(output_file)


if step == 2:
    #Step 2: Split obj folder into train and test 80-20



    train_files = int(0.8 * total_files)
    test_files = total_files - train_files
    print(f"Total files: {total_files}, Train files: {train_files}, Test files: {test_files}")

#Use shutil to copy files
    for i, file_path in enumerate(glob(os.path.join(obj_folder, "*.obj"))):
        print("Processing {}".format(file_path))
        if i < train_files:
            shutil.copy(file_path, train_folder)
        else:
            shutil.copy(file_path, test_folder)

if step == 3:
    #Make test and train labelled folders

    #Copy files from test and train to test_labelled and train_labelled
    for file_path in glob(os.path.join(test_folder, "*.obj")):
        print("Processing {}".format(file_path))
        shutil.copy(file_path, test_labelled_folder)

    for file_path in glob(os.path.join(train_folder, "*.obj")):
        print("Processing {}".format(file_path))
        shutil.copy(file_path, train_labelled_folder)

if step == 4:
    print("Run blender to add textures to train and test")

if step == 5:
    #Seperate the train labelled and test labelled folders into parts
    for input_file in glob(os.path.join(test_labelled_folder, '*.obj')):
        # get just the file name
        output_file = os.path.basename(input_file).split('.')[0]
        mesh_data = split_obj_by_color(input_file)
        for color, (vertices, indices) in mesh_data.items():
            save_as_obj(vertices, indices, test_labelled_split_folder, output_file, label[color], color)

    for input_file in glob(os.path.join(train_labelled_folder, '*.obj')):
        # get just the file name
        output_file = os.path.basename(input_file).split('.')[0]
        mesh_data = split_obj_by_color(input_file)
        for color, (vertices, indices) in mesh_data.items():
            save_as_obj(vertices, indices, train_labelled_split_folder, output_file, label[color], color)

if step == 6:
    #Convert the train and test folders into URDF
    #Convert the train_labelled_split and test_labelled_split folders into URDF
    for input_file in glob(os.path.join(train_folder, '*.obj')):

        output_file = os.path.basename(input_file).split('.')[0]
        urdf_content = generate_urdf(input_file, False)
        with open(os.path.join(urdf_folder_train, output_file + ".urdf"), "w") as f:
            f.write(urdf_content)

    for input_file in glob(os.path.join(test_folder, '*.obj')):
        output_file = os.path.basename(input_file).split('.')[0]
        urdf_content = generate_urdf(input_file, False)
        with open(os.path.join(urdf_folder_test, output_file + ".urdf"), "w") as f:
            f.write(urdf_content)

    for input_file in glob(os.path.join(train_labelled_split_folder, '*.obj')):
        output_file = os.path.basename(input_file).split('.')[0]
        urdf_content = generate_urdf(input_file, True)
        with open(os.path.join(urdf_folder_train_labelled_split, output_file + ".urdf"), "w") as f:
            f.write(urdf_content)

    for input_file in glob(os.path.join(test_labelled_split_folder, '*.obj')):
        output_file = os.path.basename(input_file).split('.')[0]
        urdf_content = generate_urdf(input_file, True)
        with open(os.path.join(urdf_folder_test_labelled_split, output_file + ".urdf"), "w") as f:
            f.write(urdf_content)

#Remember to replace mtl files with relative paths and switch forward slashes to backslashes
# python .\baselines\run_baseline.py --args_global_n_envs 15  --args_env_verbose 1  --args_baseline_dataset_type uniform --args_baseline_planner rrt_connect --args_env_randomize_ur5_pose --args_env_randomize_tree_pose  --args_callback_n_eval_orientations 600 --args_callback_n_points_per_orientation 30  --args_baseline_tree_set train --args_baseline_results_save_path rrt_connect_uniform_waypoints