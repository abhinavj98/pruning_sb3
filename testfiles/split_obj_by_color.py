import pywavefront
from collections import defaultdict
import pybullet as p
import pybullet_data

label = {
    (0.117647, 0.235294, 0.039216): "SPUR",
    (0.313725, 0.313725, 0.313725): "TRUNK",
    (0.254902, 0.176471, 0.058824): "BRANCH",
    (0.235294, 0.000000, 0.000000): "WATER_BRANCH",
}

def parse_obj(file_path):
    """Parse an OBJ file and return vertices, faces, and vertex colors."""
    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    vertices = scene.vertices
    vertices = [vertex[:3] for vertex in vertices]
    faces = scene.mesh_list[0].faces
    vertex_colors = []

    for line in open(file_path, 'r'):
        if line.startswith('v '):
            color = line.split()[4:]
            vertex_colors.append(tuple(map(float, color)))

    return vertices, faces, vertex_colors

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
        vertices_final = []

        for i, (index, vertex) in enumerate(group['vertices']):
            vertex_map[index] = i
            vertices.append(vertex)

        for face in group['faces']:
            indices.append([vertex_map[vi] for vi in face])

        mesh_data[color] = (vertices, indices)
        print(f"Color: {color}, Vertices: {len(vertices)}, Faces: {len(indices)}")

    return mesh_data

def split_obj_by_color(input_file):
    vertices, faces, vertex_colors = parse_obj(input_file)
    color_groups = group_by_color(vertices, faces, vertex_colors)
    mesh_data = create_mesh_data(color_groups)
    return mesh_data

def split_mesh(vertices, indices, max_vertices_per_chunk):
    chunks = []
    num_chunks = (len(vertices) + max_vertices_per_chunk - 1) // max_vertices_per_chunk

    for chunk_index in range(num_chunks):
        chunk_vertices = vertices[chunk_index * max_vertices_per_chunk : (chunk_index + 1) * max_vertices_per_chunk]
        chunk_indices = []

        vertex_map = {v: i for i, v in enumerate(range(chunk_index * max_vertices_per_chunk, min((chunk_index + 1) * max_vertices_per_chunk, len(vertices))))}

        for face in indices:
            if all(v in vertex_map for v in face):
                chunk_indices.append([vertex_map[v] for v in face])

        if chunk_vertices and chunk_indices:
            chunks.append((chunk_vertices, chunk_indices))

    return chunks

def load_collision_objects(mesh_data, max_vertices_per_chunk=1000):
    collision_objects = {}
    for color, (vertices, indices) in mesh_data.items():
        if len(vertices) > 0 and len(indices) > 0:
            print(f"Loading mesh for color {color}")
            print(f"Number of vertices: {len(vertices)}")
            print(f"Number of faces: {len(indices)}")

            chunks = split_mesh(vertices, indices, max_vertices_per_chunk)
            for chunk_index, (chunk_vertices, chunk_indices) in enumerate(chunks):
                print(f"Loading chunk {chunk_index} for color {color}")
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    vertices=chunk_vertices,
                    indices=[i for face in chunk_indices for i in face],
                    meshScale=[1, 1, 1]
                )
                if collision_shape_id >= 0:
                    collision_objects[f"{color}_chunk_{chunk_index}"] = collision_shape_id
                else:
                    print(f"Failed to create collision shape for color {color}, chunk {chunk_index}")
        else:
            print(f"Invalid mesh data for color {color}")
    return collision_objects

input_file = '/Users/abhinav/Desktop/gradstuff/research/pruning_sb3/meshes_and_urdf/meshes/trees/envy/train_labelled/tree_1.obj'
mesh_data = split_obj_by_color(input_file)

# Start PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

collision_objects = load_collision_objects(mesh_data)

# Example of creating a multi-body with collision shapes
for color, collision_shape_id in collision_objects.items():
    print(f"Creating multi-body for color {color}")
    id = p.createMultiBody(
        baseCollisionShapeIndex=collision_shape_id,
        basePosition=[0, 0, 0],  # Set the initial position
    )
    p.changeVisualShape(id, -1, rgbaColor=[1, 1, 1, 0])

# Run the simulation
p.setGravity(0, 0, -9.8)
while p.isConnected():
    p.stepSimulation()
