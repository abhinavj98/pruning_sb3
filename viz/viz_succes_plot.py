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

def plot_violin(df, x, label, title, val):
    plt.figure()
    sns.violinplot(x=df[x], cut=0, inner='quart')
    plt.xlabel(label)
    plt.title(title)
    if val:
        plt.axvline(val, color='red')
    plt.savefig('violin{}.png'.format(label))
#     plt.show()

if __name__ == '__main__':
    file_path = 'episode_info.csv'
    df = read_csv_file(file_path)
    df['init_point_cosine_sim_abs'] = df['init_point_cosine_sim'].abs()
    df['init_perp_cosine_sim_abs'] = df['init_perp_cosine_sim'].abs()
    df['pointing_cosine_sim_error_abs'] = df['pointing_cosine_sim_error'].abs()
    df['perpendicular_cosine_sim_error_abs'] = df['perpendicular_cosine_sim_error'].abs()

    #print total number of rows
    # # print(len(df))
    # #print success rate
    print(df['is_success'].value_counts()/len(df))
    # df = df[df['is_success'] == True]
    #print df columns
    #just keep the columns you want
    plot_success(df)
    #Calculate quat distane between ideal and start

    #Group this by success
    plot_init_conditions('init_angular_error', 'init_distance', 'Initial Angular Error', 'Initial Distance', None, None)
    plot_init_conditions('init_point_cosine_sim_abs', 'init_distance', 'Initial Pointing Cosine Similarity', 'Initial Distance', 0.7, 0)
    plot_init_conditions('init_perp_cosine_sim_abs', 'init_distance', 'Initial Perpendicular Cosine Similarity', 'Initial Distance', 0.7, 0)
    plot_init_conditions('init_point_cosine_sim_abs', 'init_perp_cosine_sim_abs', 'Initial Pointing Cosine Similarity', 'Initial Perpendicular Cosine Similarity', 0.7, 0.7)
    #sfsdfa
    plot_final_conditions('pointing_cosine_sim_error_abs', 'euclidean_error', 'Pointing Cosine Similarity',
                         'Distance', 0.7, 0)
    plot_final_conditions('perpendicular_cosine_sim_error_abs', 'euclidean_error', 'Perpendicular Cosine Similarity',
                         'Distance', 0.7, 0)
    plot_final_conditions('pointing_cosine_sim_error_abs', 'perpendicular_cosine_sim_error_abs', 'Pointing Cosine Similarity',
                         'Perpendicular Cosine Similarity', 0.7, 0.7)
    #sdfsd

    plot_success(df)
    plot_violin(df, 'euclidean_error', 'Euclidean Error', 'Final Conditions', 0.05)
    plot_violin(df, 'pointing_cosine_sim_error_abs', 'Pointing Cosine Similarity', 'Final Conditions', 0.7)
    plot_violin(df, 'perpendicular_cosine_sim_error_abs', 'Perpendicular Cosine Similarity', 'Final Conditions', 0.7)
    plot_violin(df, 'angular_error', 'Angular Error', 'Final Conditions', None)
    # import numpy as np
    # var = np.exp(-2.3)
    # b = np.random.normal(0, var, 1000)
    # #plot
    # plt.figure()
    # # sns.histplot(b, stat='count', kde=True)
    # #
    # c = np.tanh(b)/10
    # sns.histplot(c, stat='count', kde=True)
    # plt.show()