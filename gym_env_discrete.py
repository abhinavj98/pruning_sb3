
import pywavefront
import numpy as np
import random
import numpy as np
from gym import spaces
import gym
from pybullet_utils import bullet_client as bc
import os
import math
import pybullet
import pybullet_data
from collections import namedtuple
import glob

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"



class ur5GymEnv(gym.Env):
    def __init__(self,
                 renders=False,
                 maxSteps=500,
                 learning_param=0.05,
                 tree_urdf_path = None,
                 tree_obj_path = None,
                 width = 224,
                 height = 224,
                 eval = False,
                 num_points = None, 
                 action_dim = 12,
                 name = "ur5GymEnv",
                 terminate_on_singularity = True,
                 action_scale = 1,
                 movement_reward_scale = 1,
                 distance_reward_scale = 1,
                 condition_reward_scale = 1,
                 terminate_reward_scale = 1,
                 collision_reward_scale = 1,
                 slack_reward_scale = 1,
                 orientation_reward_scale = 1,
                 ):
        super(ur5GymEnv, self).__init__()
        assert tree_urdf_path != None
        assert tree_obj_path != None
        

        self.renders = renders
        self.eval = eval
        self.terminate_on_singularity = terminate_on_singularity
        self.tree_urdf_path = tree_urdf_path
        self.tree_obj_path = tree_obj_path
        self.name = name
        self.action_dim = action_dim
        self.stepCounter = 0
        self.maxSteps = maxSteps

        self.action_scale = action_scale
        self.movement_reward_scale = movement_reward_scale
        self.distance_reward_scale = distance_reward_scale
        self.condition_reward_scale = condition_reward_scale
        self.terminate_reward_scale = terminate_reward_scale
        self.collision_reward_scale = collision_reward_scale
        self.slack_reward_scale = slack_reward_scale
        self.orientation_reward_scale = orientation_reward_scale

        # setup pybullet sim:
        if self.renders:
            self.con = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.con = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.con.setTimeStep(5./240.)
        self.con.setGravity(0,0,-10)
        self.con.setRealTimeSimulation(False)
      
        self.con.resetDebugVisualizerCamera( cameraDistance=1.06, cameraYaw=-120.3, cameraPitch=-12.48, cameraTargetPosition=[-0.3,-0.06,0.4])
      
        
        self.observation_space = spaces.Dict({ 
            'depth': spaces.Box(low=-1.,
                     high=1.0,
                     shape=(1, 224, 224), dtype=np.float32),\
            'goal_pos': spaces.Box(low = -5.,
                        high = 5.,
                        shape = (3,), dtype=np.float32),
            'cur_pos': spaces.Box(low = -5.,
                        high = 5.,
                        shape = (3,), dtype=np.float32),
            'cur_or': spaces.Box(low = -2,
                        high = 2,
                        shape = (4,), dtype=np.float32),
            'joint_angles' : spaces.Box(low = -2*np.pi,
                        high = 2*np.pi,
                        shape = (6,), dtype=np.float32),
            'joint_velocities' : spaces.Box(low = -6,
                        high = 6,
                        shape = (6,), dtype=np.float32),
                        })
        
        self.reset_counter = 0
        self.randomize_tree_count = 5
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_dim,), dtype=np.float32)
        self.learning_param = learning_param
 
        # setup robot arm:
        self.setup_ur5_arm()
        self.reset_env_variables()
        #Camera parameters
        self.near_val = 0.01
        self.far_val = 3
        self.height = height
        self.width = width
        self.proj_mat = self.con.computeProjectionMatrixFOV(
            fov=42, aspect = width / height, nearVal=self.near_val,
            farVal=self.far_val)

        
        #Tree parameters
        self.tree_goal_pos = [1, 0, 0] # initial object pos
        self.tree_goal_branch = [0, 0, 0]

        self.trees = Tree.make_list_from_folder(self, self.tree_urdf_path, self.tree_obj_path, pos = np.array([0, -0.8, 0]), orientation=np.array([0,0,0,1]), scale=0.1, num_points = num_points)
        self.tree = random.sample(self.trees, 1)[0]
        self.tree.active()

        #Debug parameters
        self.debug_line = -1
        self.debug_cur_or = -1
        self.debug_des_or = -1
        self.debug_branch = -1
       
    def reset_env_variables(self):
        # Env variables that will change
        self.observation = {}
        self.stepCounter = 0
        self.terminated = False
        self.singularity_terminated = False
        self.collisions = 0
        


    def setup_ur5_arm(self):
        self.end_effector_index = 7
        flags = self.con.URDF_USE_SELF_COLLISION
        self.ur5 = self.con.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = self.con.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
        
        self.joints = dict()
        for i in range(self.num_joints):
            info = self.con.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                self.con.setJointMotorControl2(self.ur5, info.id, self.con.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
    
        self.init_joint_angles = (-1.57, -1.57,1.80,-3.14,-1.57, -1.57)
        self.set_joint_angles(self.init_joint_angles)
        for i in range(1000):
            self.con.stepSimulation()
        self.init_pos = self.con.getLinkState(self.ur5, self.end_effector_index)
        self.joint_velocities = [0,0,0,0,0,0]
        self.joint_angles = self.init_joint_angles
        self.achieved_pos = self.get_current_pose()[0]
        self.previous_pose = (np.array(0), np.array(0))

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        self.con.setJointMotorControlArray(
            self.ur5, indexes,
            self.con.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )

    def set_joint_velocities(self, joint_velocities):
       #Set joint velocities using pybullet motor control
        velocities = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        maxForce = 500
        self.con.setJointMotorControlArray(self.ur5, 
        indexes, 
        controlMode=self.con.VELOCITY_CONTROL,
        targetVelocities = joint_velocities,
       )

    def calculate_joint_velocities_from_end_effector_velocity(self, end_effector_velocity):
        #Calculate joint velocities from end effector velocity
        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0,0,0], self.get_joint_angles(), [0,0,0,0,0,0], [0,0,0,0,0,0])
        jacobian = np.vstack(jacobian)
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity)
        return joint_velocities

    def get_joint_angles(self):
        j = self.con.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints


    def check_collisions(self):
        collisions = self.con.getContactPoints(bodyA = self.ur5)#, linkIndexA=self.end_effector_index)
        for i in range(len(collisions)):
            # print("collision")
            if collisions[i][-6] < 0 :
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True
        return False


    def calculate_ik(self, position, orientation):
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
       
        joint_angles = self.con.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, orientation,
            jointDamping=[0.01]*6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges
        )
        return joint_angles


    def get_current_pose(self):
        linkstate = self.con.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[4], linkstate[1] #Position wrt end effector, orientation wrt COM
        return (position, orientation)

    def set_camera(self, pose, orientation):
        pose, orientation = self.get_current_pose()
        rot_mat = np.array(self.con.getMatrixFromQuaternion(orientation)).reshape(3,3)
		#Initial vectors
        init_camera_vector = np.array([1, 0, 0])#
        init_up_vector = np.array([0, 0,1]) #
        #Rotated vectors
        camera_vector = rot_mat.dot(init_camera_vector)
        up_vector = rot_mat.dot(init_up_vector)
        view_matrix = self.con.computeViewMatrix(pose, pose + 0.1 * camera_vector, up_vector)
       
        return self.con.getCameraImage(self.width, self.height, viewMatrix = view_matrix, projectionMatrix = self.proj_mat, renderer = self.con.ER_BULLET_HARDWARE_OPENGL)
        
    @staticmethod
    def seperate_rgbd_rgb_d(rgbd, h = 224, w = 224):
        rgb = rgbd[2][:,:,0:3].reshape(3,h,w)/255
        depth = rgbd[3]
        return rgb, depth

    @staticmethod
    def linearize_depth(depth, far_val, near_val):
        depth_linearized = near_val / (far_val - (far_val -near_val) * depth + 0.00000001)
        return depth_linearized
   
    def get_rgbd_at_cur_pose(self):
        cur_p = self.get_current_pose()
        rgbd = self.set_camera(cur_p[0], cur_p[1])
        rgb,  depth = self.seperate_rgbd_rgb_d(rgbd)
        depth = depth.astype(np.float32)
        depth = self.linearize_depth(depth, self.far_val, self.near_val) - 0.5
        return rgb, depth

    def reset(self):

        self.reset_env_variables()
        self.reset_counter+=1
        #Remove and add tree to avoid collisions with tree while resetting
        self.tree.inactive()
        self.con.removeBody(self.ur5)
        #Remove debug items
        self.con.removeUserDebugItem(self.debug_branch)
        self.con.removeUserDebugItem(self.debug_line)
        self.con.removeUserDebugItem(self.debug_cur_or)
        self.con.removeUserDebugItem(self.debug_des_or)
        #Sample new tree if reset_counter is a multiple of randomize_tree_count
        if self.reset_counter%self.randomize_tree_count == 0:
            self.tree = random.sample(self.trees, 1)[0]

        #Create new ur5 arm body
        self.setup_ur5_arm()
        
        # Sample new point
        random_point = random.sample(self.tree.reachable_points,1)[0]
        self.tree_goal_pos = random_point[0]
        self.tree_goal_branch = random_point[1]
        self.tree.active()

        #Add debug branch
        self.debug_branch = self.con.addUserDebugLine(self.tree_goal_pos - 50*self.tree_goal_branch,self.tree_goal_pos+ 50*self.tree_goal_branch, [1,0,0], 200)
      
        self.getExtendedObservation()
      
        return self.observation

    def step(self, action):
        # remove debug line
        self.con.removeUserDebugItem(self.debug_line)
        self.previous_pose = self.get_current_pose()
        self.prev_joint_velocities = self.joint_velocities

        action = action*self.action_scale
        self.joint_velocities = self.calculate_joint_velocities_from_end_effector_velocity(action)
        self.set_joint_velocities(self.joint_velocities)
        
        for i in range(1):
            self.con.stepSimulation()
            # if self.renders: time.sleep(5./240.) 
        self.getExtendedObservation()
        reward, reward_infos = self.compute_reward(self.achieved_pos, self.achieved_or, self.desired_pos, self.previous_pos, None)
        self.debug_line = self.con.addUserDebugLine(self.achieved_pos, self.desired_pos, [0,0,1], 20)
        done, terminate_info = self.is_task_done()
        
        infos = {'is_success': False}
        if terminate_info['time_limit_exceeded']:
            infos["TimeLimit.truncated"] = True
            infos["terminal_observation"] = self.observation
        
        if self.terminated == True:
            infos['is_success'] = True
        
        self.stepCounter += 1
        infos['episode'] = {"l": self.stepCounter,  "r": reward}
        infos.update(reward_infos)
        return self.observation, reward, done, infos

    def render(self, mode = "human"):
        cam_prop = (1024, 768)
        img_rgbd = self.con.getCameraImage(cam_prop[0], cam_prop[1])
        return img_rgbd[2]
     
    def close(self):
        self.con.disconnect()


    def getExtendedObservation(self):
        """
        The observations are the current position, the goal position, the current orientation, the current depth image, the current joint angles and the current joint velocities    
        """   
        tool_pos, tool_orient = self.get_current_pose()
        goal_pos = self.tree_goal_pos

        self.achieved_pos = np.array(tool_pos).astype(np.float32) 
        self.observation['cur_pos'] = np.array(tool_pos).astype(np.float32) - np.array(self.init_pos[0]).astype(np.float32)

        self.desired_pos = np.array(goal_pos).astype(np.float32)
        self.observation['goal_pos'] = np.array(goal_pos).astype(np.float32) - np.array(self.init_pos[0]).astype(np.float32)

        self.previous_pos = np.array(self.previous_pose[0])
        self.previous_or = np.array(self.previous_pose[1])

        self.achieved_or = np.array(tool_orient).astype(np.float32)
        self.observation['cur_or'] = self.achieved_or

        self.rgb, self.depth = self.get_rgbd_at_cur_pose()
        self.observation['depth'] = np.expand_dims(self.depth.astype(np.float32), axis = 0)

        self.joint_angles = np.array(self.get_joint_angles()).astype(np.float32)
        self.observation['joint_angles'] = np.array(self.joint_angles).astype(np.float32) - self.init_joint_angles
        self.observation['joint_velocities'] = np.array(self.joint_velocities).astype(np.float32)

    def is_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        time_limit_exceeded = self.stepCounter > self.maxSteps
        singularity_achieved = self.singularity_terminated
        goal_achieved = self.terminated
        c = (self.singularity_terminated == True or self.terminated == True or self.stepCounter > self.maxSteps)
        terminate_info = {"time_limit_exceeded": time_limit_exceeded, "singularity_achieved": singularity_achieved, "goal_achieved": goal_achieved}
        return c, terminate_info

    def get_condition_number(self):
        #get jacobian
        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0,0,0], self.get_joint_angles(), [0,0,0,0,0,0], [0,0,0,0,0,0])
        jacobian = np.vstack(jacobian)
        condition_number = np.linalg.cond(jacobian)
        return condition_number
    
    def compute_orientation_reward(self, achieved_pos, desired_pos, achieved_or, branch_vector):
        # Orientation reward is computed as the dot product between the current orientation and the perpedicular vector to the end effector and goal pos vector
        # This is to encourage the end effector to be perpendicular to the branch

        #Perpendicular vector to branch vector
        perpendicular_vector = compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector+desired_pos)

        #Get vector for current orientation of end effector
        rot_mat = np.array(self.con.getMatrixFromQuaternion(achieved_or)).reshape(3,3)
		#Initial vectors
        init_vector = np.array([1, 0, 0])
        camera_vector = rot_mat.dot(init_vector)
        self.con.removeUserDebugItem(self.debug_cur_or)
        self.con.removeUserDebugItem(self.debug_des_or)
        self.debug_des_or = self.con.addUserDebugLine(achieved_pos, achieved_pos + perpendicular_vector, [1,0,0], 2)
        self.debug_cur_or = self.con.addUserDebugLine(self.achieved_pos, self.achieved_pos + 0.1 * camera_vector, [0, 1, 0], 1)
       
        orientation_reward = np.dot(camera_vector, perpendicular_vector)/(np.linalg.norm(camera_vector)*np.linalg.norm(perpendicular_vector))*self.orientation_reward_scale
        return orientation_reward
       
    
    def compute_reward(self, achieved_pos, achieved_or, desired_pos, previous_pos, info):
        reward = float(0)
        reward_info = {}
        # Give rewards better names, and appropriate scales

        self.collisions = 0

        self.delta_movement = float(goal_reward(achieved_pos, previous_pos, desired_pos))
        self.target_dist = float(goal_distance(achieved_pos, desired_pos))

        movement_reward = self.delta_movement*self.movement_reward_scale
        reward_info['movement_reward'] = movement_reward
        reward += movement_reward

        distance_reward = (np.exp(-self.target_dist*5)*self.distance_reward_scale)
        reward_info['distance_reward'] = distance_reward
        reward += distance_reward

        orientation_reward = np.exp(self.compute_orientation_reward(achieved_pos, desired_pos, achieved_or, self.tree_goal_branch)*3)/np.exp(3)*self.orientation_reward_scale
        #Mostly within 0.8 to 1
        #Compress to increase range

        reward_info['orientation_reward'] = orientation_reward
        reward += orientation_reward
        # print('Orientation reward: ', orientation_reward)
        # camera_vector = camera_vector/np.linalg.norm(camera_vector)
        # perpendicular_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)
       
        # print('Orientation reward: ', orientation_reward, np.arccos(camera_vector[0])*180/np.pi - np.arccos(perpendicular_vector[0])*180/np.pi, np.arccos(camera_vector[1])*180/np.pi - np.arccos(perpendicular_vector[1])*180/np.pi, np.arccos(camera_vector[2])*180/np.pi - np.arccos(perpendicular_vector[2])*180/np.pi)
      
      
        
        condition_number = self.get_condition_number()
        condition_number_reward = 0
        if condition_number > 50 or (self.joint_velocities > 5).any():
            print('Too high condition number!')
            self.singularity_terminated = True
            condition_number_reward = -3
        elif self.terminate_on_singularity:
            condition_number_reward = np.abs(1/condition_number)*self.condition_reward_scale
        reward += condition_number_reward
        reward_info['condition_number_reward'] = condition_number_reward

        terminate_reward = 0
        if self.target_dist < self.learning_param and orientation_reward > 0.95*self.orientation_reward_scale:  # and approach_velocity < 0.05:
            self.terminated = True
            terminate_reward = 1*self.terminate_reward_scale
            reward += terminate_reward
            print('Successful!')
        reward_info['terminate_reward'] = terminate_reward
        
        # check collisions:
        collision_reward = 0
        if self.check_collisions():
            collision_reward = 1*self.collision_reward_scale
            self.collisions+=1

        reward += collision_reward
        reward_info['collision_reward'] = collision_reward
        
        slack_reward = 1*self.slack_reward_scale
        reward_info['slack_reward'] = slack_reward
        reward+= slack_reward

        #Minimize joint velocities
        velocity_mag = np.linalg.norm(self.joint_velocities)
        velocity_reward = -np.clip(velocity_mag, -0.1, 0.1)
        #reward += velocity_rewarid
        reward_info['velocity_reward'] = velocity_reward
        return reward, reward_info



def goal_distance(goal_a, goal_b):
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def goal_reward(current, previous, target):
    # Compute the reward between the previous and current goal.
    assert current.shape == previous.shape
    assert current.shape == target.shape
    diff_prev = goal_distance(previous, target)
    diff_curr = goal_distance(current, target)
    reward = diff_prev - diff_curr
    return reward

# x,y distance
def goal_distance2d(goal_a, goal_b):
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

def compute_perpendicular_projection(a, b, c):
    ab = b - a
    bc = c - b
    projection = ab - np.dot(ab, bc)/np.dot(bc, bc) * bc
    return projection

def compute_perpendicular_projection_vector(ab, bc):
    projection = ab - np.dot(ab, bc)/np.dot(bc, bc) * bc
    return projection

class Tree():
    def __init__(self, env, urdf_path, obj_path, pos = np.array([0,0,0]), orientation = np.array([0,0,0,1]), num_points = None, scale = 1) -> None:
        self.urdf_path = urdf_path
        self.env = env
        self.scale = scale
        self.pos = pos
        self.orientation = orientation
        self.tree_obj = pywavefront.Wavefront(obj_path, create_materials = True, collect_faces=True)
        self.vertex_and_projection = []
        self.transformed_vertices = list(map(self.transform_obj_vertex, self.tree_obj.vertices))
        self.projection_mean = 0
      
        #Find the two longest edges of the face
        #Add their mid-points and perpendicular projection to the smallest side as a point and branch
        for face in self.tree_obj.mesh_list[0].faces:
            #Order the sides of the face by length
            ab = (face[0], face[1], np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[1]]))
            ac = (face[0], face[2], np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[2]]))
            bc = (face[1], face[2], np.linalg.norm(self.transformed_vertices[face[1]] - self.transformed_vertices[face[2]]))
            sides = [ab, ac, bc]
            #argsort sorts in ascending order
            sorted_sides = np.argsort([x[2] for x in sides])
            ac = sides[sorted_sides[2]]
            ab = sides[sorted_sides[1]]
            bc = sides[sorted_sides[0]]
            #|a
            #|\
            #| \
            #|  \
            #|   \
            #|    \
            #b______\c
            perpendicular_projection = compute_perpendicular_projection_vector(self.transformed_vertices[ac[0]] - self.transformed_vertices[ac[1]], self.transformed_vertices[bc[0]] - self.transformed_vertices[bc[1]])
                                                    
            self.vertex_and_projection.append(((self.transformed_vertices[ac[0]]+self.transformed_vertices[ac[1]])/2, perpendicular_projection))
            self.vertex_and_projection.append(((self.transformed_vertices[ab[0]]+self.transformed_vertices[ab[1]])/2, perpendicular_projection))
            self.projection_mean += np.linalg.norm(perpendicular_projection)
            self.projection_mean += np.linalg.norm(perpendicular_projection)
        self.projection_mean = self.projection_mean/len(self.vertex_and_projection)
        self.num_points = num_points
        self.get_reachable_points(self.env.ur5)
        
    def active(self):
        self.tree_urdf = self.env.con.loadURDF(self.urdf_path, self.pos, self.orientation, globalScaling=self.scale)

    def inactive(self):
        self.env.con.removeBody(self.tree_urdf)

    def transform_obj_vertex(self, vertex):
        vertex_pos = np.array(vertex[0:3])*self.scale
        vertex_orientation = [0,0,0,1] #Dont care about orientation
        vertex_w_transform = self.env.con.multiplyTransforms(self.pos, self.orientation, vertex_pos, vertex_orientation)
        return np.array(vertex_w_transform[0])

    def is_reachable(self, vertice, ur5):
        ur5_base_pos = np.array(self.env.con.getBasePositionAndOrientation(ur5)[0])
        dist=np.linalg.norm(ur5_base_pos - vertice[0], axis=-1)
        projection_length = np.linalg.norm(vertice[1])
        if dist >= 1 or projection_length < self.projection_mean* 0.7:
            return False
        j_angles = self.env.calculate_ik(vertice[0], None)
        self.env.set_joint_angles(j_angles)
        self.env.con.stepSimulation()
        ee_pos, _ = self.env.get_current_pose()
        dist=np.linalg.norm(np.array(ee_pos) - vertice[0], axis=-1)
        condition_number = self.env.get_condition_number()
        if dist <= 0.05 and condition_number < 20: 
            return True
        return False

    def get_reachable_points(self, ur5):
        self.reachable_points = list(filter(lambda x: self.is_reachable(x, ur5), self.vertex_and_projection))
        # self.reachable_points = [np.array(i[0][0:3]) for i in self.reachable_points]
        np.random.shuffle(self.reachable_points)
        if self.num_points:
            self.reachable_points = self.reachable_points[0:self.num_points]
        print("Number of reachable points: ", len(self.reachable_points))
       
        return 

    @staticmethod
    def make_list_from_folder(env, trees_urdf_path, trees_obj_path, pos, orientation, scale, num_points):
        trees = []
        for urdf, obj in zip(sorted(glob.glob(trees_urdf_path+'/*.urdf')), sorted(glob.glob(trees_obj_path+'/*.obj'))):
            trees.append(Tree(env, urdf_path=urdf, obj_path=obj, pos=pos, orientation = orientation, scale=scale, num_points=num_points))
            break
        return trees
 