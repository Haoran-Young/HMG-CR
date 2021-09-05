import numpy as np
import pickle
from tqdm import tqdm
import torch
import time
import os
from torch_geometric.data import Data


class BL_Contrarec():
    def __init__(self, dataset, max_num_buy, train_negative_samples, batch_size):
        self.dataset_path = './data/' + dataset + '/'
        self.max_num_buy = max_num_buy
        self.train_negative_samples = train_negative_samples
        self.batch_size = batch_size

    def _load_data(self):
        print('... Loading Data Dicts ...')
        with open(self.dataset_path + 'train_data_dict.pkl', 'rb') as f:
            self.train_data_dict = pickle.load(f)
        with open(self.dataset_path + 'val_test_dict.pkl', 'rb') as f:
            self.val_test_dict = pickle.load(f)
        with open(self.dataset_path + 'user_item_cat_num.pkl', 'rb') as f:
            self.user_num, self.item_num, self.cat_num = pickle.load(f)

    def _get_max_interactions_length(self):
        self.max_behaviors_length = -1
        self.max_items_length = -1
        self.max_cats_length = -1

        #interactions_len = []

        print('... Acquiring Max Interactions Length ...')
        for i in tqdm(range(1, self.user_num + 1)):
            behaviors_length = len(self.train_data_dict[i]['behaviors'])
            #interactions_len.append(behaviors_length)
            if behaviors_length > self.max_behaviors_length:
                self.max_behaviors_length = behaviors_length

            items_length = len(np.unique(self.train_data_dict[i]['items']))
            if items_length > self.max_items_length:
                self.max_items_length = items_length

            cats_length = len(np.unique(self.train_data_dict[i]['cats']))
            if cats_length > self.max_cats_length:
                self.max_cats_length = cats_length

        # interactions_len.sort()
        # print(interactions_len[-50:-1])
        
    def _load_train_targets(self):
        if os.path.exists(self.dataset_path + 'contrarec_train_data.pkl'):
            return

        self.train_targets = []
        self.train_labels = []

        print('... Loading Train Targets ...')
        for i in tqdm(range(1, self.user_num + 1)):
            train_data_i = self.train_data_dict[i]
            buy_index = np.where(train_data_i['behaviors'] == 4)[0]
            if len(buy_index) > self.max_num_buy:
                buy_index = buy_index[-self.max_num_buy:]
            buy_items = train_data_i['items'][buy_index]

            train_targets_i = np.append(buy_items, train_data_i['negative_samples'][: self.train_negative_samples * len(buy_items)])
            train_targets_i = np.append(train_targets_i, [0 for j in range((1 + self.train_negative_samples) * self.max_num_buy - len(train_targets_i))])

            train_labels_i = np.array([-1 for j in range(len(train_targets_i))])
            train_labels_i[: len(buy_items)] = [1 for j in range(len(buy_items))]
            train_labels_i[len(buy_items): (1 + self.train_negative_samples) * len(buy_items)] = [0 for j in range(self.train_negative_samples * len(buy_items))]

            self.train_targets.append(train_targets_i)
            self.train_labels.append(train_labels_i)

    def _load_val_test_targets(self):
        if os.path.exists(self.dataset_path + 'contrarec_train_data.pkl'):
            return
            
        self.val_targets = []
        self.test_targets = []

        print('... Loading Val & Test Targets ...')
        for i in tqdm(range(1, self.user_num + 1)):
            val_targets_i = np.append(self.val_test_dict[i]['val_target'], self.val_test_dict[i]['negative_samples'])
            val_targets_i = np.append(val_targets_i, [0 for j in range((1 + self.train_negative_samples) * self.max_num_buy - 100)])
            
            test_targets_i = np.append(self.val_test_dict[i]['test_target'], self.val_test_dict[i]['negative_samples'])
            test_targets_i = np.append(test_targets_i, [0 for j in range((1 + self.train_negative_samples) * self.max_num_buy - 100)])
            
            self.val_targets.append(val_targets_i)
            self.test_targets.append(test_targets_i)

    def _load_graphs(self):
        self.graphs_1 = []
        self.graphs_2 = []
        self.graphs_3 = []
        self.graphs_4 = []

        print('... Loading Training Graphs ...')
        if os.path.exists(self.dataset_path + 'contrarec_train_data.pkl'):
            with open(self.dataset_path + 'contrarec_train_data.pkl', 'rb') as f:
                contrarec_train_data = pickle.load(f)
                self.graphs_1 = contrarec_train_data['graphs_1']
                self.graphs_2 = contrarec_train_data['graphs_2']
                self.graphs_3 = contrarec_train_data['graphs_3']
                self.graphs_4 = contrarec_train_data['graphs_4']
        else:
            for i in tqdm(range(1, self.user_num + 1)):
                user_id = [i]
                train_data_dict_i = self.train_data_dict[i]

                behavior_ids = [0 for j in range(self.max_behaviors_length)]
                behavior_ids[: len(train_data_dict_i['behaviors'])] = train_data_dict_i['behaviors']

                item_ids = [0 for j in range(self.max_items_length)]
                unique_item_ids = np.unique(train_data_dict_i['items'])
                item_ids[: len(unique_item_ids)] = unique_item_ids

                cat_ids = [0 for j in range(self.max_cats_length)]
                unique_cat_ids = np.unique(train_data_dict_i['cats'])
                cat_ids[: len(unique_cat_ids)] = unique_cat_ids

                id = np.concatenate((user_id, behavior_ids, item_ids, cat_ids))

                graph_1 = np.matrix(np.zeros([len(id), len(id)]))
                graph_2 = np.matrix(np.zeros([len(id), len(id)]))
                graph_3 = np.matrix(np.zeros([len(id), len(id)]))
                graph_4 = np.matrix(np.zeros([len(id), len(id)]))

                graph_1[0, np.where(np.array(behavior_ids) == 4)[0] + 1] = 1

                graph_2[0, np.where(np.array(behavior_ids) == 1)[0] + 1] = 1
                graph_2[0, np.where(np.array(behavior_ids) == 4)[0] + 1] = 1

                graph_3[0, np.where(np.array(behavior_ids) != 3)[0] + 1] = 1

                graph_4[0, 1: len(behavior_ids) + 1] = np.ones([1, len(behavior_ids)])

                def b_b_interaction(graph):
                    b_i_graph = graph[1: len(behavior_ids) + 1, 1: len(behavior_ids) + len(item_ids) + 1 ]
                    b_i_graph_T = b_i_graph.T
                    b_b_interaction_matrix = np.dot(b_i_graph, b_i_graph_T)
                    graph[1: len(behavior_ids) + 1, 1: len(behavior_ids) + 1] = np.triu(b_b_interaction_matrix, k=1)
                    return graph

                def graph_data_generator(user_id, graph_id, graph, id):
                    edge_index = torch.LongTensor(np.where(graph != 0))
                    if graph_id == '4':
                        x = torch.LongTensor(id.reshape(-1, 1))
                        val_test_labels = np.array([0 for j in range((1 + self.train_negative_samples) * self.max_num_buy)])
                        val_test_labels[0] = 1
                        y = [self.train_targets[user_id], self.train_labels[user_id], self.val_targets[user_id], val_test_labels, self.test_targets[user_id], val_test_labels]
                        y = torch.LongTensor(y)
                        graph_data = Data(x=x, edge_index=edge_index, y=y)
                    else:
                        graph_data = Data(edge_index=edge_index)
                    return graph_data

                for j in range(len(train_data_dict_i['behaviors'])):
                    item_id = train_data_dict_i['items'][j]
                    item_id_index = unique_item_ids.tolist().index(item_id)
                    cat_id = train_data_dict_i['cats'][j]
                    cat_id_index = unique_cat_ids.tolist().index(cat_id)

                    if train_data_dict_i['behaviors'][j] == 4:
                        graph_1[1 + j, 1 + len(behavior_ids) + item_id_index] = 1
                    if train_data_dict_i['behaviors'][j] == 4 or train_data_dict_i['behaviors'][j] == 1:
                        graph_2[1 + j, 1 + len(behavior_ids) + item_id_index] = 1
                    if train_data_dict_i['behaviors'][j] != 3:
                        graph_3[1 + j, 1 + len(behavior_ids) + item_id_index] = 1
                    graph_4[1 + j, 1 + len(behavior_ids) + item_id_index] = 1

                    graph_1[1 + len(behavior_ids) + item_id_index, 1 + len(behavior_ids) + len(item_ids) + cat_id_index] = 1
                    graph_2[1 + len(behavior_ids) + item_id_index, 1 + len(behavior_ids) + len(item_ids) + cat_id_index] = 1
                    graph_3[1 + len(behavior_ids) + item_id_index, 1 + len(behavior_ids) + len(item_ids) + cat_id_index] = 1
                    graph_4[1 + len(behavior_ids) + item_id_index, 1 + len(behavior_ids) + len(item_ids) + cat_id_index] = 1

                graph_1 = b_b_interaction(graph_1)
                graph_2 = b_b_interaction(graph_2)
                graph_3 = b_b_interaction(graph_3)
                graph_4 = b_b_interaction(graph_4)

                graph_data_1 = graph_data_generator(i-1, '1', graph_1, id)
                graph_data_2 = graph_data_generator(i-1, '2', graph_2, id)
                graph_data_3 = graph_data_generator(i-1, '3', graph_3, id)
                graph_data_4 = graph_data_generator(i-1, '4', graph_4, id)

                self.graphs_1.append(graph_data_1)
                self.graphs_2.append(graph_data_2)
                self.graphs_3.append(graph_data_3)
                self.graphs_4.append(graph_data_4)

            with open(self.dataset_path + 'contrarec_train_data.pkl', 'wb') as f:
                pickle.dump({'graphs_1': self.graphs_1, 'graphs_2': self.graphs_2, 'graphs_3': self.graphs_3, 'graphs_4': self.graphs_4}, f)

    def load(self):
        self._load_data()
        self._get_max_interactions_length()
        self._load_train_targets()
        self._load_val_test_targets()
        self._load_graphs()