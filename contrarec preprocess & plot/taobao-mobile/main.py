import pandas as pd
import time

def read_csv(path):
    df = pd.read_csv(path)
    return df

def convert_behavior(df):
    df.iloc[df.iloc[:, 2] == 1, 2] = 'pv'
    df.iloc[df.iloc[:, 2] == 2, 2] = 'fav'
    df.iloc[df.iloc[:, 2] == 3, 2] = 'cart'
    df.iloc[df.iloc[:, 2] == 4, 2] = 'buy'

def delete_user_geo(df):
    df = df.drop(columns='user_geohash')
    return df

def exchange_order(df):
    cats_id = df.iloc[:, 3]
    df = df.drop(columns='item_category')
    df.insert(2, 'item_category', cats_id)
    return df

def time_map(t):
    timeArray = time.strptime(t, '%Y-%m-%d %H')
    timestamp = time.mktime(timeArray)
    return timestamp

def convert_time(df):
    df.iloc[:, 4] = df.iloc[:, 4].map(time_map)
    return df

def main():
    df = read_csv('UserBehavior.csv')
    convert_behavior(df)
    df = delete_user_geo(df)
    df = exchange_order(df)
    df = convert_time(df)
    df.to_csv('modified_UserBehavior.csv', index=0)

if __name__ == '__main__':
    main()