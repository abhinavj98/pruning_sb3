import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
def plot_violin(df, row_name, label, title, tree_type, val=None, y= None, save=False, xlim=None, scatter=False):
    """
    Plots a violin plot for a specific column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        row_name (str): The column name to plot.
        label (str): Label for the x-axis.
        title (str): Title of the plot.
        val (float, optional): A vertical line value to highlight on the plot.
        save (bool, optional): Whether to save the plot as a file.
        xlim (list, optional): Limits for the x-axis.
    """
    plt.figure()
    plt.tight_layout()

    # Define the color palette
    palette = [(0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.466, 0.674, 0.188)]

    # Create the violin plot
    sns.violinplot(data=df, x=row_name, cut=0, inner='quart', orient='h', palette=palette).set(yticklabels=[])
    plt.xlabel(label)
    plt.title(title + f' ({tree_type})')

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    if scatter:
        sns.stripplot(data=df, x=row_name, y=y, color='black', size=5, jitter=0., alpha=0.4)
    # Add a vertical line if val is provided
    if val is not None:
        plt.axvline(val, color=palette[2], linestyle='--')

    # Save or show the plot
    if save:
        plt.savefig(f'violin_{row_name}_{tree_type}.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()



if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--tree_type", type=str, default="ufo", help="Type of tree to analyze")
    args = args.parse_args()
    tree_type = args.tree_type
    final_summary = []
    result_files = glob.glob(f'results_cleaned/{tree_type}*.csv')
    for result_file in result_files:
        df = pd.read_csv(result_file)

        if 'type' not in df.columns or len(df) < 2:
            continue

        desired = df[df['type'] == 'd'].sort_values(by='timestamp')
        achieved = df[df['type'] == 'a'].sort_values(by='timestamp')

        min_len = min(len(desired), len(achieved))
        if min_len == 0:
            continue

        desired = desired.iloc[:min_len]
        achieved = achieved.iloc[:min_len]
        print(desired, achieved)

        # Positions
        desired_pos = desired[['x', 'y', 'z']].values
        achieved_pos = achieved[['x', 'y', 'z']].values

        # Euclidean position error
        euclidean_error = np.linalg.norm(desired_pos - achieved_pos, axis=1)

        # Orientations
        desired_q = desired[['qx', 'qy', 'qz', 'qw']].values
        achieved_q = achieved[['qx', 'qy', 'qz', 'qw']].values

        try:
            desired_rot = R.from_quat(desired_q)
            achieved_rot = R.from_quat(achieved_q)

            # X-axis pointing difference
            cosine_similarity_x = np.sum(desired_rot.as_matrix()[:, 0] * achieved_rot.as_matrix()[:, 0], axis=1)
            cosine_similarity_x = np.clip(cosine_similarity_x, -1.0, 1.0)
            error_x = np.arccos(cosine_similarity_x)

            # Z-axis pointing difference
            cosine_similarity_z = np.sum(desired_rot.as_matrix()[:, 2] * achieved_rot.as_matrix()[:, 2], axis=1)
            cosine_similarity_z= np.clip(cosine_similarity_z, -1.0, 1.0)
            error_z = np.arccos(cosine_similarity_z)

            # Full rotation error (rotation difference magnitude)
            rotation_diff = desired_rot.inv() * achieved_rot
            rotvecs = rotation_diff.as_rotvec()
            full_rot_error = np.linalg.norm(rotvecs, axis=1)

            # Store file-level mean errors
            final_summary.append({
                'file': os.path.basename(result_file),
                'group': os.path.basename(result_file).split('_')[0],
                'euclidean_error_mean_m': np.mean(euclidean_error),
                'euclidean_error_std_m': np.std(euclidean_error),
                'x_axis_error_mean_deg': np.degrees(np.mean(error_x)),
                'z_axis_error_mean_deg': np.degrees(np.mean(error_z)),
                'full_rotation_error_mean_deg': np.degrees(np.mean(full_rot_error)),
                'cosine_similarity_x_mean': np.mean(cosine_similarity_x),
                'cosine_similarity_z_mean': np.mean(cosine_similarity_z),
                'success': np.sum(euclidean_error < 0.08 and np.degrees(np.mean(error_z)) < 34.37 and np.degrees(np.mean(error_x)) < 34.37) / len(euclidean_error),
                'distance_from_base': np.mean(np.linalg.norm(desired_pos, axis=1)),
                'within_reachable_space': np.mean(np.linalg.norm(desired_pos, axis=1)) > 0.8 and np.mean(np.linalg.norm(desired_pos, axis=1)) < 1.1 and desired['y'] < -0.75,
                'desired_y': np.mean(desired['y']),
            })
        except ValueError:
            continue

    # Save into a clean DataFrame
    final_df = pd.DataFrame(final_summary)
    plot_violin(final_df, 'euclidean_error_mean_m', 'Euclidean Error (m)', 'Euclidean Error', tree_type, 0.08, save=True, scatter=True, xlim=[-0., 0.2])
    plot_violin(final_df, 'z_axis_error_mean_deg', 'Pointing Error (deg)', 'Pointing Error',tree_type, 30, save=True, scatter=True, xlim=[-0., 60])

    plot_violin(final_df, 'x_axis_error_mean_deg', 'Perpendicular Error (deg)', 'Perpendicular Error',tree_type, 30, save=True, scatter=True, xlim=[-0., 60])
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Full Cosine and Euclidean Errors by File", dataframe=final_df)

    print(final_df)