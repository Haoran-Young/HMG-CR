import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import time


class DataLoader():
    def __init__(self, dataset, min_num_buy, train_negative_samples):
        self.dataset_path = './data/' + dataset + '/'
        self.min_num_buy = min_num_buy
        self.train_negative_samples = train_negative_samples

    def _is_data_exist(self):
        return os.path.exists(self.dataset_path + 'data_dict.pkl')

    def _get_statistics_file(self):
        df = pd.read_csv(self.dataset_path + 'UserBehavior.csv', header=None)
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

        with open(self.dataset_path + 'statistics.pkl', 'wb') as f:
            pickle.dump(users_behaviors, f)

    def _load_statistics_file(self):
        print('... Loading Data Statistics ...')
        
        with open(self.dataset_path+'statistics.pkl', 'rb') as f:
            self.statistics = np.array(pickle.load(f))

    def _data_filtering(self):
        print('... Data Filtering ...')
        
        filtered_statistics = self.statistics[self.statistics[:, 3] >= self.min_num_buy, :]
        filtered_statistics = filtered_statistics[filtered_statistics[:, 0] <= 800, :]
        self.filtered_users = filtered_statistics[:, 4]

        raw_df = pd.read_csv(self.dataset_path + 'UserBehavior.csv', header=None)
        self.filtered_df = raw_df[raw_df.iloc[:, 0].isin(self.filtered_users)]

    def _id_mapping(self, id_name):
        if id_name == 'user_id':
            print('... Generating User Index Mapping ...')
            id_list = np.array(self.filtered_df.iloc[:, 0])
        elif id_name == 'item_id':
            print('... Generating Item Index Mapping ...')
            id_list = np.array(self.filtered_df.iloc[:, 1])
        elif id_name == 'cat_id':
            print('... Generating Category Index Mapping ...')
            id_list = np.array(self.filtered_df.iloc[:, 2])

        id_list = np.unique(id_list)
        id_list.sort()
        id_list = id_list.tolist()
        id_list_len = len(id_list)
        new_id_list = [i for i in range(1, id_list_len + 1)]
        id_mapping_dict = dict(zip(id_list, new_id_list))

        return id_mapping_dict

    def _data_unification(self):
        print('... Data Unifying ...')

        self.filtered_df.iloc[:, 1] = self.filtered_df.iloc[:, 1].map(lambda x: self._item_id_mapping_dict[x])
        self.filtered_df.iloc[:, 2] = self.filtered_df.iloc[:, 2].map(lambda x: self._cat_id_mapping_dict[x])
        self.filtered_df.iloc[:, 3] = self.filtered_df.iloc[:, 3].map(lambda x: self._behavior_mapping_dict[x])

        self.data_dict = dict()
        for user in tqdm(self.filtered_users):
            user_interactions = self.filtered_df[self.filtered_df.iloc[:, 0] == user]
            user_interactions = np.array(user_interactions)
            items = user_interactions[:, 1].reshape(-1)
            cats = user_interactions[:, 2].reshape(-1)
            behaviors = user_interactions[:, 3].reshape(-1)
            timestamps = user_interactions[:, 4].reshape(-1)
            user_dict = {'behaviors': behaviors, 'timestamps': timestamps, 'items': items, 'cats': cats}
            self.data_dict[self._user_id_mapping_dict[user]] = user_dict

    def load(self):
        if self._is_data_exist():
            print('... Data Dict Exists ...')
            with open(self.dataset_path + 'data_dict.pkl', 'rb') as f:
                self.data_dict = pickle.load(f)
            with open(self.dataset_path + 'user_item_cat_num.pkl', 'rb') as f:
                self.user_num, self.item_num, self.cat_num = pickle.load(f)

            # user_num = len(self.data_dict)
            # behaviors = []
            # items = []
            # for user in self.data_dict.keys():
            #     behaviors.extend(self.data_dict[user]['behaviors'])
            #     items.extend(self.data_dict[user]['items'])
            # behaviors = np.array(behaviors)
            # items_num = len(set(items))

            # print(user_num)
            # print(items_num)
            # print(np.count_nonzero(behaviors == 1))
            # print(np.count_nonzero(behaviors == 2))
            # print(np.count_nonzero(behaviors == 3))
            # print(np.count_nonzero(behaviors == 4))

            # time.sleep(50)
        else:
            if not os.path.exists(self.dataset_path + 'statistics.pkl'):
                self._get_statistics_file()
            self._load_statistics_file()
            self._data_filtering()

            self._user_id_mapping_dict = self._id_mapping('user_id')
            self._item_id_mapping_dict = self._id_mapping('item_id')
            self._cat_id_mapping_dict = self._id_mapping('cat_id')
            self._behavior_mapping_dict = {'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4}

            self.user_num = len(self._user_id_mapping_dict)
            self.item_num = len(self._item_id_mapping_dict)
            self.cat_num = len(self._cat_id_mapping_dict)

            self._data_unification()

            with open(self.dataset_path + 'data_dict.pkl', 'wb') as f:
                pickle.dump(self.data_dict, f)
            with open(self.dataset_path + 'user_item_cat_num.pkl', 'wb') as f:
                pickle.dump([self.user_num, self.item_num, self.cat_num], f)


    def train_validation_test_data_generator(self):
        if os.path.exists(self.dataset_path + 'train_data_dict.pkl'):
            print('... Training Set Exists ...')
            with open(self.dataset_path + 'train_data_dict.pkl', 'rb') as f:
                self.train_data_dict = pickle.load(f)
            with open(self.dataset_path + 'val_test_dict.pkl', 'rb') as f:
                self.val_test_dict = pickle.load(f)
            
        else:
            print('... Training Set Generating ...')
            self.train_data_dict = dict()
            self.val_test_dict = dict()

            for i in tqdm(range(1, len(self.data_dict) + 1)):
                self.train_data_dict[i] = dict()
                self.val_test_dict[i] = dict()

                each_data = self.data_dict[i]
                buy_index = np.where(each_data['behaviors'] == 4)[0]

                self.train_data_dict[i]['behaviors'] = np.delete(each_data['behaviors'], buy_index[-2:])
                self.train_data_dict[i]['timestamps'] = np.delete(each_data['timestamps'], buy_index[-2:])
                self.train_data_dict[i]['items'] = np.delete(each_data['items'], buy_index[-2:])
                self.train_data_dict[i]['cats'] = np.delete(each_data['cats'], buy_index[-2:])

                buy_items = each_data['items'][buy_index]
                items_list = np.array([j for j in range(1, self.item_num + 1)])
                train_negative_samples = np.random.choice(np.setdiff1d(items_list, buy_items), len(buy_items) * self.train_negative_samples, replace=False)
                test_negative_samples = np.random.choice(np.setdiff1d(np.setdiff1d(items_list, buy_items), train_negative_samples), 99, replace=False)
                
                self.train_data_dict[i]['negative_samples'] = train_negative_samples
                self.val_test_dict[i]['val_target'] = buy_items[-2]
                self.val_test_dict[i]['test_target'] = buy_items[-1]
                self.val_test_dict[i]['negative_samples'] = test_negative_samples

            with open(self.dataset_path + 'train_data_dict.pkl', 'wb') as f:
                pickle.dump(self.train_data_dict, f)
            with open(self.dataset_path + 'val_test_dict.pkl', 'wb') as f:
                pickle.dump(self.val_test_dict, f)