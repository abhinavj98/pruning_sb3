import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#read csv file
def read_csv_file(file_name):
    df = pd.read_csv(file_name)
    return df

def plot_success(df):
    sns.scatterplot(x=df['pointx'], y=df['pointz'], hue=df['is_success'])
    plt.show()

def plot_init_conditions(x, y):
    temp_df = df[[x,y]]
    sns.histplot(
        temp_df, x=x, y=y,
        stat='count', cbar = True)
    plt.show()

def plot_violin(df, x):
    sns.violinplot(x=df[x], cut=0)
    plt.show()

if __name__ == '__main__':
    file_path = '../reward.csv'
    df = read_csv_file(file_path)
    df['is_success'][0] = True
    df['init_point_cosine_sim_abs'] = df['init_point_cosine_sim'].abs()
    df['init_perp_cosine_sim_abs'] = df['init_perp_cosine_sim'].abs()
    df['pointing_cosine_sim_error_abs'] = df['pointing_cosine_sim_error'].abs()
    df['perpendicular_cosine_sim_error_abs'] = df['perpendicular_cosine_sim_error'].abs()

    #print df columns
    #just keep the columns you want
    # plot_success(df)
    #Calculate quat distane between ideal and start
    # plot_init_conditions('init_point_cosine_sim_abs', 'init_distance')
    # plot_init_conditions('init_perp_cosine_sim_abs', 'init_distance')
    # plot_init_conditions('init_point_cosine_sim_abs', 'init_perp_cosine_sim_abs')
    print(df)
    plot_violin(df, 'euclidean_error')
    plot_violin(df, 'pointing_cosine_sim_error_abs')
    plot_violin(df, 'perpendicular_cosine_sim_error_abs')