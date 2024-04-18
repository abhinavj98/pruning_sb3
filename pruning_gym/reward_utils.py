import numpy as np
from typing import List, Tuple
from nptyping import NDArray, Shape, Float
from pybullet import getMatrixFromQuaternion, getDifferenceQuaternion, getQuaternionFromEuler


class Reward:
    def __init__(self, movement_reward_scale, distance_reward_scale, pointing_orientation_reward_scale,
                 perpendicular_orientation_reward_scale, terminate_reward_scale, collision_reward_scale,
                 slack_reward_scale, condition_reward_scale):
        self.movement_reward_scale = movement_reward_scale
        self.distance_reward_scale = distance_reward_scale
        self.pointing_orientation_reward_scale = pointing_orientation_reward_scale
        self.perpendicular_orientation_reward_scale = perpendicular_orientation_reward_scale
        self.terminate_reward_scale = terminate_reward_scale
        self.collision_reward_scale = collision_reward_scale
        self.slack_reward_scale = slack_reward_scale
        self.condition_reward_scale = condition_reward_scale
        self.velocity_reward_scale = 0.
        self.reward_info = {}

    def calculate_perpendicular_orientation_reward(self, achieved_or,
                                                 previous_or,
                                                 branch_vector):
        cosine_sim_perp_prev = self.compute_perpendicular_cos_sim(previous_or, branch_vector)
        cosine_sim_perp = self.compute_perpendicular_cos_sim(achieved_or, branch_vector)
        perpendicular_orientation_reward = (abs(cosine_sim_perp) - abs(cosine_sim_perp_prev)) * \
                                              self.perpendicular_orientation_reward_scale
        self.reward_info['perpendicular_orientation_reward'] = perpendicular_orientation_reward
        return perpendicular_orientation_reward, abs(cosine_sim_perp)



    def calculate_movement_reward(self, achieved_pos, previous_pos, desired_pos):
        # Compute the reward between the previous and current goal.
        assert achieved_pos.shape == desired_pos.shape
        assert achieved_pos.shape == previous_pos.shape
        diff_curr = np.linalg.norm(achieved_pos - desired_pos)
        diff_prev = np.linalg.norm(previous_pos - desired_pos)
        movement_reward = (diff_prev - diff_curr) * self.movement_reward_scale
        self.reward_info['movement_reward'] = movement_reward
        return movement_reward

    def calculate_distance_reward(self, achieved_pos, desired_pos):
        # Compute the reward between the previous and current goal.
        assert achieved_pos.shape == desired_pos.shape
        distance_reward = -np.linalg.norm(achieved_pos - desired_pos) * self.distance_reward_scale
        self.reward_info['distance_reward'] = distance_reward
        return distance_reward

    def calculate_pointing_orientation_reward(self, achieved_pos, desired_pos, achieved_or, previous_pos,
                                              previous_or, branch_vector):
        cosine_sim_prev = self.compute_pointing_cos_sim(previous_pos, desired_pos, previous_or, branch_vector)
        cosine_sim_curr = self.compute_pointing_cos_sim(achieved_pos, desired_pos, achieved_or, branch_vector)
        #if sign of both are different, then it means that the end effector has crossed the branch
        pointing_orientation_reward = (cosine_sim_curr - cosine_sim_prev) * \
                                      self.pointing_orientation_reward_scale
        self.reward_info['pointing_orientation_reward'] = pointing_orientation_reward
        return pointing_orientation_reward, abs(cosine_sim_curr)



    def calculate_condition_number_reward(self, condition_number):
        condition_number_reward = np.abs(1 / condition_number) * self.condition_reward_scale
        self.reward_info['condition_number_reward'] = condition_number_reward
        return condition_number_reward

    def calculate_termination_reward(self, terminated):
        terminate_reward = int(terminated) * self.terminate_reward_scale
        self.reward_info['terminate_reward'] = terminate_reward
        return terminate_reward

    def calculate_acceptable_collision_reward(self, collision_info):
        collision_reward = int(collision_info['collisions_acceptable']) * self.collision_reward_scale
        self.reward_info['collision_acceptable_reward'] = collision_reward
        return collision_reward

    def calculate_unacceptable_collision_reward(self, collision_info):
        collision_reward = int(collision_info['collisions_unacceptable']) * self.collision_reward_scale
        self.reward_info['collision_unacceptable_reward'] = collision_reward
        return collision_reward

    def calculate_slack_reward(self):
        slack_reward = 1 * self.slack_reward_scale
        self.reward_info['slack_reward'] = slack_reward
        return slack_reward

    def calculate_velocity_minimization_reward(self, velocity):
        velocity_reward = -np.linalg.norm(velocity) * self.velocity_reward_scale
        self.reward_info['velocity_reward'] = velocity_reward
        self.reward_info['velocity'] = np.linalg.norm(velocity)
        return velocity_reward


    """
    
            ^ z
            |
     x <----|
            / Out of screen
            y
            Pybullet coords
    """
    @staticmethod
    def compute_perpendicular_cos_sim(achieved_or,
                                      branch_vector):
        # Orientation reward is computed as the dot product between the current orientation and the perpendicular
        # vector to the end effector and goal pos vector This is to encourage the end effector to be perpendicular to
        # the branch
        # Perpendicular vector to branch vector
        # Get vector for current orientation of end effector
        rot_mat = np.array(getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        # Initial vectors
        init_vector = np.array([1, 0, 0]) #Coz of starting orientation of end effector
        camera_vector = rot_mat.dot(init_vector)
        # Check antiparallel case as well
        cosine_sim_perp = np.dot(camera_vector, branch_vector) / (
                np.linalg.norm(camera_vector) * np.linalg.norm(branch_vector))
        return cosine_sim_perp

    @staticmethod
    def compute_perpendicular_projection(a: NDArray[Shape['3, 1'], Float], b: NDArray[Shape['3, 1'], Float],
                                         c: NDArray[Shape['3, 1'], Float]):
        ab = b - a
        bc = c - b
        projection = ab - np.dot(ab, bc) / np.dot(bc, bc) * bc
        return projection

    @staticmethod
    def compute_pointing_cos_sim(achieved_pos,
                                 desired_pos,
                                 achieved_or,
                                 branch_vector):
        # Orientation reward is computed as the dot product between the current orientation and the
        # perpendicular vector to the end effector and goal pos vector
        # This is to encourage the end effector to be perpendicular to the branch

        # Perpendicular vector to branch vector
        perpendicular_vector = Reward.compute_perpendicular_projection(achieved_pos, desired_pos,
                                                                     branch_vector + desired_pos)
        rot_mat = np.array(getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        # Initial vectors
        init_vector = np.array([0, 0, 1]) #Coz of starting orientation of end effector
        camera_vector = rot_mat.dot(init_vector)
        pointing_cos_sim = np.dot(camera_vector, perpendicular_vector) / (
                np.linalg.norm(camera_vector) * np.linalg.norm(perpendicular_vector))
        return pointing_cos_sim

    @staticmethod
    def compute_distance_quaternions(q1, q2):
        q1_2 = getDifferenceQuaternion(q1, q2)
        theta = 2 * np.arccos(np.abs(q1_2[3]))
        return theta

    @staticmethod
    def get_angular_distance_to_goal(current_or_mat, branch_vector, achieved_pos, desired_pos):
        #get pointing vector
        #red branch
        #blue poitinh
        #TODO: Decide on system for rot mat
        pointing_vector = Reward.compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector + desired_pos)
        pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)
        #get perpendicular vector
        perpendicular_vector = -branch_vector
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        up_vector = -np.cross(perpendicular_vector, pointing_vector)
        #get rotation matrix
        rf = np.array([perpendicular_vector, up_vector, pointing_vector])
        # print(rf)
        #compute difference in rotation matrix
        diff = np.matmul(current_or_mat, rf.T)
        #get theta
        theta = np.arccos((np.trace(diff) - 1) / 2)

        return theta, rf
