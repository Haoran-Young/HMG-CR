import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

df = pd.read_csv('./data/taobao/UserBehavior.csv', header=None)
users = list(set(df.iloc[:, 0].tolist()))

users_behaviors = []
for user in tqdm(users):
    behaviors_statistics = {'pv': 0, 'cart': 0, 'fav': 0, 'buy': 0}
    user_behaviors = df[df.iloc[:, 0] == user]
    user_behaviors_num = user_behaviors.iloc[:, 3].value_counts()
    for each in ['pv', 'cart', 'fav', 'buy']:
        try:
            each_num = user_behaviors_num[each]
            behaviors_statistics[each] = each_num
        except:
            break
    user_behavior = list(behaviors_statistics.values())
    user_behavior.append(user)
    users_behaviors.append(user_behavior)

with open('./data/taobao/statistics.pkl', 'wb') as f:
    pickle.dump(users_behaviors, f)