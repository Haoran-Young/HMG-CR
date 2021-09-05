import pandas as pd
from tqdm import *
import time

def read_csv(path):
    df = pd.read_csv(path)
    return df

def total_user_sort(df):
    df = df.sort_values(by='user_id')
    return df

def inner_user_sort(df):
    user_dfs = []
    user_ids = list(set(df.iloc[:, 0]))
    user_ids.sort()
    with trange(len(user_ids)) as t:
        for i in t:
            user_df = df[df.iloc[:, 0] == user_ids[i]].copy()
            user_df.sort_values(by='time_stamp', inplace=True)
            user_dfs.append(user_df)
    df = pd.concat(user_dfs, ignore_index=True)

    return df

def main():
    df = read_csv('modified_UserBehavior.csv')
    df = total_user_sort(df)
    df = inner_user_sort(df)
    df.to_csv('sorted_UserBehavior.csv', header=0, index=0)

if __name__ == '__main__':
    main()