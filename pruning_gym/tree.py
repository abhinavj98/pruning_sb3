import glob
import os
import pickle
from typing import Optional, Tuple, List
import pybullet
import numpy as np
import pywavefront
from nptyping import NDArray, Shape, Float

# from pruning_sb3.pruning_gym.pruning_env import SUPPORT_AND_POST_PATH
from pruning_sb3.pruning_gym.helpers import compute_perpendicular_projection_vector

from pruning_sb3.pruning_gym import MESHES_AND_URDF_PATH, ROBOT_URDF_PATH, SUPPORT_AND_POST_PATH
from pruning_sb3.pruning_gym.reward_utils import Reward
# from memory_profiler import profile
class Tree:
    """ Class representing a tree mesh. It is used to sample points on the surface of the tree."""

    def __init__(self, env, pyb, urdf_path: str, obj_path: str,
                 pos: NDArray[Shape['3,1'], Float] = np.array([0, 0, 0]),
                 orientation: NDArray[Shape['4,1'], Float] = np.array([0, 0, 0, 1]),
                 num_points: Optional[int] = None, scale: int = 1, curriculum_distances: Tuple = (-0.1,),
                 curriculum_level_steps: Tuple = (), reset_count = 0) -> None:


        assert len(curriculum_distances) - 1 == len(curriculum_level_steps)

        self.urdf_path = urdf_path
        self.env = env
        self.pyb = pyb
        self.scale = scale
        self.pos = pos
        self.orientation = orientation
        self.obj_path = obj_path
        self.tree_obj = pywavefront.Wavefront(obj_path, create_materials=True, collect_faces=True)
        self.vertex_and_projection = []
        self.projection_mean = np.array(0.)
        self.projection_std = np.array(0.)
        self.projection_sum_x = np.array(0.)
        self.projection_sum_x2 = np.array(0.)
        self.base_xyz = self.env.ur5.get_current_pose(self.env.ur5.base_index)[0]
        self.num_points = num_points
        self.reachable_points = []
        self.curriculum_points = dict()
        self.curriculum_distances = curriculum_distances
        self.curriculum_level_steps = curriculum_level_steps
        self.reset_count = reset_count
        self.supports = None
        self.tree_id = None

        #Filter trunk points?
        # self.filtered_vertices = list(filter(self.filter_trunk_points, self.tree_obj.vertices))
        self.transformed_vertices = list(map(self.transform_obj_vertex, self.tree_obj.vertices))

        # if pickled file exists load and return
        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        # TODO: Add reset variable so that even if present it recomputes
        if not os.path.exists('./pkl/' + str(path_component[3])):
            os.makedirs('./pkl/' + str(path_component[3]))
        pkl_path = './pkl/' + str(path_component[3]) + '/' + str(path_component[-1][:-5]) + '_reachable_points.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.pos = data[0]
                self.orientation = data[1]
                self.reachable_points = data[2]
            print('Loaded reachable points from pickle file ', pkl_path)
            print("Number of reachable points: ", len(self.reachable_points))
        else:
            self.get_all_points()
            self.get_reachable_points()
            # dump reachable points to file using pickle

            with open(pkl_path, 'wb') as f:
                pickle.dump((self.pos, self.orientation, self.reachable_points), f)

            print('Saved reachable points to pickle file ', self.urdf_path[:-5] + '_reachable_points.pkl')
            print("Number of reachable points: ", len(self.reachable_points))
        self.make_curriculum()


        # if self.curriculum_distances:
        #     self.make_curriculum(self.curriculum_distances)

        # Find the two longest edges of the face
        # Add their mid-points and perpendicular projection to the smallest side as a point and branch

    def transform_obj_vertex(self, vertex: List) -> NDArray[Shape['3, 1'], Float]:
        vertex_pos = np.array(vertex[0:3]) * self.scale
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation
        vertex_w_transform = self.pyb.con.multiplyTransforms(self.pos, self.orientation, vertex_pos, vertex_orientation)
        return np.array(vertex_w_transform[0])
    def filter_trunk_points(self, vertex) -> bool:
        vertex_pos = np.array(vertex[0:3])
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation
        # Tree specific condition
        if "envy" in self.urdf_path:
            if abs(vertex_pos[0]-self.pos[0]) < 0.05:
                return False
        return True

    def get_all_points(self):
        for num, face in enumerate(self.tree_obj.mesh_list[0].faces):
            # Order the sides of the face by length
            ab = (
                face[0], face[1],
                np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[1]]))
            ac = (
                face[0], face[2],
                np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[2]]))
            bc = (
                face[1], face[2],
                np.linalg.norm(self.transformed_vertices[face[1]] - self.transformed_vertices[face[2]]))

            normal_vec = np.cross(self.transformed_vertices[ac[0]] - self.transformed_vertices[ac[1]],
                                  self.transformed_vertices[bc[0]] - self.transformed_vertices[bc[1]])
            # Only front facing faces
            if np.dot(normal_vec, [0, 1, 0]) < 0:
                continue
            # argsort sorts in ascending order
            sides = [ab, ac, bc]
            sorted_sides = np.argsort([x[2] for x in sides])
            ac = sides[sorted_sides[2]]
            ab = sides[sorted_sides[1]]
            bc = sides[sorted_sides[0]]
            # |a
            # |\
            # | \
            # |  \
            # |   \
            # |    \
            # b______\c
            perpendicular_projection = compute_perpendicular_projection_vector(
                self.transformed_vertices[ac[0]] - self.transformed_vertices[ac[1]],
                self.transformed_vertices[bc[0]] - self.transformed_vertices[bc[1]])

            scale = np.random.uniform()
            tree_point = ((1-scale)*self.transformed_vertices[ab[0]] + scale*self.transformed_vertices[ab[1]])
            self.vertex_and_projection.append((tree_point, perpendicular_projection,
                 normal_vec))

            # This projection mean is used to filter corner/flushed faces which do not correspond to a branch
            self.projection_sum_x += np.linalg.norm(perpendicular_projection)
            self.projection_sum_x2 += np.linalg.norm(perpendicular_projection) ** 2
        self.projection_mean = self.projection_sum_x / len(self.vertex_and_projection)
        self.projection_std = np.sqrt(
            self.projection_sum_x2 / len(self.vertex_and_projection) - self.projection_mean ** 2)
    def active(self):
        print("activating tree")
        assert self.tree_id is None
        assert self.supports is None
        print('Loading tree from ', self.urdf_path)
        self.supports = self.pyb.con.loadURDF(SUPPORT_AND_POST_PATH, [0, -0.6, 0],
                                              list(self.pyb.con.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])),
                                              globalScaling=1)
        self.tree_id = self.pyb.con.loadURDF(self.urdf_path, self.pos, self.orientation, globalScaling=self.scale)

    def inactive(self):
        assert self.tree_id is not None
        assert self.supports is not None
        self.pyb.con.removeBody(self.tree_id)
        self.pyb.con.removeBody(self.supports)
        self.tree_id = None
        self.supports = None



    def is_reachable(self, vertice: Tuple[NDArray[Shape['3, 1'], Float], NDArray[Shape['3, 1'], Float]]) -> bool:
        #TODO: Fix this
        ur5_base_pos = np.array(self.base_xyz)


        #Meta condition
        dist = np.linalg.norm(ur5_base_pos - vertice[0], axis=-1)
        projection_length = np.linalg.norm(vertice[1])
        if dist >= 0.8 or (projection_length < self.projection_mean + 0.5 * self.projection_std):
            # print("proj length")
            return False

        j_angles = self.env.ur5.calculate_ik(vertice[0], None)
        self.env.ur5.set_joint_angles(j_angles)
        for i in range(100):
            self.pyb.con.stepSimulation()
        ee_pos, _ = self.env.ur5.get_current_pose(self.env.ur5.end_effector_index)
        dist = np.linalg.norm(np.array(ee_pos) - vertice[0], axis=-1)
        if dist <= 0.05:
            return True

        return False

    def get_reachable_points(self) -> List[Tuple[NDArray[Shape['3, 1'], Float], NDArray[Shape['3, 1'], Float]]]:
        self.reachable_points = list(filter(lambda x: self.is_reachable(x), self.vertex_and_projection))
        #Filter points on trunk
        self.reachable_points = list(filter(lambda x: self.filter_trunk_points(x[0]), self.reachable_points))
        # self.reachable_points = [np.array(i[0][0:3]) for i in self.reachable_points]
        np.random.shuffle(self.reachable_points)
        if self.num_points:
            self.reachable_points = self.reachable_points[0:self.num_points]
        print("Number of reachable points: ", len(self.reachable_points))
        if len(self.reachable_points) < 1:
            print("No points in reachable points", self.urdf_path)
            # self.reset_tree()

        return self.reachable_points

    @staticmethod
    def make_trees_from_folder(env, pyb, trees_urdf_path: str, trees_obj_path: str, pos: NDArray,
                               orientation: NDArray, scale: int, num_points: int, num_trees: int,
                               curriculum_distances: Tuple, curriculum_level_steps: Tuple):
        trees: List[Tree] = []
        for urdf, obj in zip(sorted(glob.glob(trees_urdf_path + '/*.urdf')),
                             sorted(glob.glob(trees_obj_path + '/*.obj'))):
            if len(trees) >= num_trees:
                break
            #randomize position TOOO:
            randomize = True
            if randomize:
                pos = pos + np.random.uniform(low = -1, high=1, size = (3,)) * np.array([0.25, 0.025, 0.25])
                # pos[2] = pos[2] - 0.3
                orientation = np.array([0,0,0,1])#pybullet.getQuaternionFromEuler(np.random.uniform(low = -1, high=1, size = (3,)) * np.pi / 180 * 10)
            trees.append(Tree(env, pyb, urdf_path=urdf, obj_path=obj, pos=pos, orientation=orientation, scale=scale,
                              num_points=num_points, curriculum_distances=curriculum_distances,
                              curriculum_level_steps=curriculum_level_steps))
        return trees


    def reset_tree(self):
        self.reset_count += 1
        if self.reset_count > 2:
            print("Reset count exceeded for tree ", self.urdf_path)
            return
        self.pos = np.array([0,0,0.6]) + np.random.uniform(low = -1, high=1, size = (3,)) * np.array([0.2, 0.1, .2])
        self.pos[2] = self.pos[2] - 2
        orientation = pybullet.getQuaternionFromEuler(np.random.uniform(low = -1, high=1, size = (3,)) * np.pi / 180 * 15)
        self.__init__(self.env, self.pyb, urdf_path=self.urdf_path, obj_path=self.obj_path, pos=self.pos, orientation=orientation, scale=self.scale,
                              num_points=self.num_points, curriculum_distances=self.curriculum_distances,
                              curriculum_level_steps=self.curriculum_level_steps, reset_count=self.reset_count)

        self.make_curriculum()
        # self.tree = Tree(env, pyb_con, urdf_path=urdf, obj_path=obj, pos=pos, orientation=orientation, scale=scale,
        #                       num_points=num_points, curriculum_distances=curriculum_distances,
        #                       curriculum_level_steps=curriculum_level_steps))

    def make_curriculum(self, init_or = None):
        """This function is used to create a curriculum of points on the tree. The curriculum is created by
        sampling points on the tree and then filtering them based on distance from the base of the arm and the
        perpendicular cosine sim of the end effector orientation and the branch vector."""
        # self.env.ur5.remove_ur5_robot()
        # self.env.ur5.setup_ur5_arm()
        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        #Allow pkling and loading
        if not os.path.exists('./pkl5/' + str(path_component[3])):
            os.makedirs('./pkl5/' + str(path_component[3]))
        for level, max_distance in enumerate(self.curriculum_distances):
            self.curriculum_points[level] = []
            # pkl_path = './pkl3/' + str(path_component[3]) + '/' + str(
            #     path_component[-1][:-5]) + '_curriculum_{}.pkl'.format(distance)
            #
            # if os.path.exists(pkl_path):
            #     with open(pkl_path, 'rb') as f:
            #         curriculum_points = pickle.load(f)
            #     self.curriculum_points[level] = [(distance, i) for i in curriculum_points]
            #     print('Loaded curriculum points from pickle file ',
            #           pkl_path)
            #     print("Number of curriculum points: ", len(self.curriculum_points[level]))
            # else:
            for point in self.reachable_points:
                target_dist = np.linalg.norm(np.array(self.base_xyz) - point[0], axis=-1)
                pcs = Reward.compute_perpendicular_cos_sim(self.env.ur5.init_pos[1], point[1])
                if abs(pcs) > 0.7:
                    continue
                if target_dist < max_distance:
                    self.curriculum_points[level].append((max_distance, point))

                # with open(pkl_path, 'wb') as f:
                #     pickle.dump(self.curriculum_points[level], f)

            # if pkl path exists else create
            if len(self.curriculum_points[level]) < 1:
                print("No points in curriculum level ", level)
                # self.reset_tree()

            print("Curriculum level: ", level, "Number of points: ", len(self.curriculum_points[level]))
            # Memory management
            del self.transformed_vertices
            del self.vertex_and_projection
            del self.reachable_points