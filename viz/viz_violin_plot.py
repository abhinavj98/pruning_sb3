import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
#read csv file
def read_csv_file(file_name):
    df = pd.read_csv(file_name)
    return df

def plot_success(df):
    #Plot x,y and see if point is successfull
    #Extend to include 4 categories, success_rl, success_rrt, success_both, success_none
    #create figure
    plt.figure()
    sns.scatterplot(x=df['pointx'], y=df['pointz'], hue=df['is_success'])
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Success")
    #save plot
    plt.savefig('success.png')
    # plt.show()

def plot_init_conditions(x, y, x_label, y_label, x_val, y_val):
    plt.figure()
    temp_df = df[[x,y]]
    sns.histplot(
        temp_df, x=x, y=y,
        stat='count', cbar = True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Initial Conditions")
    #save plot
    if x_val:
        plt.axvline(x_val, color='red')
    if y_val:
        plt.axhline(y_val, color='red')
    plt.savefig('init_conditions{}{}.png'.format(x_label, y_label))

def plot_final_conditions(x, y, x_label, y_label, x_val, y_val):
    plt.figure()
    temp_df = df[[x,y]]
    sns.histplot(
        temp_df, x=x, y=y,
        stat='count', cbar = True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Final Conditions")
    #save plot
    if x_val:
        plt.axvline(x_val, color='red')
    if y_val:
        plt.axhline(y_val, color='red')
    plt.savefig('final_conditions{}{}.png'.format(x_label, y_label))
#     plt.show()

def plot_violin(df, x, label, title, val, save = False, y = None, xlim = None):
    plt.figure()
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.1)

    # palette = ['#0072B2', '#009E73', '#D55E00']
    palette = [(0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.466, 0.674, 0.188)]
    sns.violinplot(data=df, x=x, cut=0, inner='quart', orient='h', y=y, hue = y, split=True, gap = -0.2, palette=palette, legend = 'brief').set(yticklabels=[])
    plt.xlabel(label)
    plt.ylabel(' ')
    if xlim:
        plt.xlim(xlim)
    #scatter plot where Environment is real world
    #opacity = 0.5
    sns.stripplot(data=df[df['Environment'] == 'Real World'], x=x, y=y, color='black', size=5, jitter = 0., alpha = 0.4)
    if val:
        plt.axvline(val, color=palette[2])
    if save:
        plt.savefig('violin{}.png'.format(label),  bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()

def plot_density(df, x, label, title, val):
    plt.figure()
    sns.kdeplot(data = df, x=df[x], y = "count", cut=0, shade = True)
    plt.xlabel(label)
    plt.title(title)
    if val:
        plt.axvline(val, color='red')
    plt.show()
    # plt.savefig('density{}.png'.format(label))

if __name__ == '__main__':
    file_path = 'results_data/policy_uniform.csv'
    real_world_file_path = 'results_data/real_world_results_transformed.csv'
    save = True
    df = read_csv_file(file_path)
    df_real = read_csv_file(real_world_file_path)
    df['pointing_cosine_angle_error_abs'] = np.arccos(df['pointing_cosine_sim_error']).abs()
    df['perpendicular_cosine_angle_error_abs'] = np.arccos(df['perpendicular_cosine_sim_error']).abs()
    data1 = df_real['euclidean_error']
    data2 = df['euclidean_error']
    df_euc = pd.DataFrame({
        'Euclidean Error': np.concatenate([data1, data2]),
        'Environment': ['Real World'] * len(data1) + ['Simulation'] * len(data2)
    })
    plot_violin(df_euc, 'Euclidean Error', 'Euclidean Error (m)', 'Euclidean Error', 0.05, save, 'Environment', xlim=[-0., 0.25])

    data1 = df_real['perpendicular_cosine_angle_error_abs']
    data2 = df['perpendicular_cosine_angle_error_abs']
    df_perp = pd.DataFrame({
        'Perpendicular angle error': np.concatenate([data1, data2]),
        'Environment': ['Real World'] * len(data1) + ['Simulation'] * len(data2)
    })
    plot_violin(df_perp, 'Perpendicular angle error', 'Perpendicular angle error (rad)', 'Euclidean Error', 0.52, save, 'Environment', xlim=[-0.1, 1.75])

    data1 = df_real['pointing_cosine_angle_error_abs']
    data2 = df['pointing_cosine_angle_error_abs']
    df_point = pd.DataFrame({
        'Pointing angle error': np.concatenate([data1, data2]),
        'Environment': ['Real World'] * len(data1) + ['Simulation'] * len(data2)
    })
    plot_violin(df_point, 'Pointing angle error', 'Pointing angle error (rad)', 'Euclidean Error', 0.52, save, 'Environment', xlim=[-0.1, 1.75])
    #get rows where all conditions are satisfied
    df_success = df_real[(df_real['euclidean_error'] < 0.05) & (df_real['pointing_cosine_angle_error_abs'] < 0.52) & (df_real['perpendicular_cosine_angle_error_abs'] < 0.52)]
    print(len(df_success)/len(df_real))