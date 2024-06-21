import numpy as np
import pandas as pd

# read csv file
df = pd.read_csv('results_data/real_world_results.csv')
goal_euclidean = np.array(df[['xg', 'yg', 'zg']])
finish_euclidean = np.array(df[['xf', 'yf', 'zf']])
# get euclidean distance between goal and finish
euclidean_error = np.linalg.norm(goal_euclidean - finish_euclidean, axis=1)
print(euclidean_error)

goal_orientation = np.array(df[['ag', 'bg', 'cg', 'wg']])
finish_orientation = np.array(df[['af', 'bf', 'cf', 'wf']])
# convert quaternions to rotation matrices using scipy
from scipy.spatial.transform import Rotation as R

goal_rotation = R.from_quat(goal_orientation)
finish_rotation = R.from_quat(finish_orientation)
# get cosine similarity between goal and finish x axis
cosine_similarity_perp = np.sum(goal_rotation.as_matrix()[:, 0] * finish_rotation.as_matrix()[:, 0], axis=1)
error_perp = np.arccos(cosine_similarity_perp)
# get cosine similarity between goal and finish y axis
cosine_similarity_point = np.sum(goal_rotation.as_matrix()[:, 1] * finish_rotation.as_matrix()[:, 1], axis=1)
error_point = np.arccos(cosine_similarity_point)
df['euclidean_error'] = euclidean_error
df['pointing_cosine_angle_error_abs'] = error_point
df['perpendicular_cosine_angle_error_abs'] = error_perp
# save to csv
df.to_csv('results_data/real_world_results_transformed.csv')
