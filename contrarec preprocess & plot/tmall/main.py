import pandas as pd
import time

def read_csv(path):
    df = pd.read_csv(path)
    return df

def convert_behavior(df):
    df.iloc[df.iloc[:, 6] == 0, 6] = 'pv'
    df.iloc[df.iloc[:, 6] == 1, 6] = 'cart'
    df.iloc[df.iloc[:, 6] == 2, 6] = 'buy'
    df.iloc[df.iloc[:, 6] == 3, 6] = 'fav'

def delete_user_geo(df):
    df = df.drop(columns='seller_id')
    df = df.drop(columns='brand_id')
    return df

def exchange_order(df):
    actions = df.iloc[:, 4]
    df = df.drop(columns='action_type')
    df.insert(3, 'action_type', actions)
    return df

def main():
    df = read_csv('UserBehavior.csv')
    convert_behavior(df)
    df = delete_user_geo(df)
    df = exchange_order(df)
    df.to_csv('modified_UserBehavior.csv', index=0)

if __name__ == '__main__':
    main()