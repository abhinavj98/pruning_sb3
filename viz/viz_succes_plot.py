import pandas as pd
import seaborn as sns


#read csv file
def read_csv_file(file_name):
    df = pd.read_csv(file_name)
    return df

if __name__ == '__main__':
    file_path = '../reward.csv'
    df = read_csv_file(file_path)
    print(df['pointx'], df['pointy'], df['is_success'])