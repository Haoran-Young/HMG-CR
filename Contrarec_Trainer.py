import os
import pickle
from torch_geometric.data import DataLoader 
import random
from Contrarec_model import Contrarec_model
import torch
import torch.optim as optim
from tqdm import *
import time
from Evaluator import Evaluator
import numpy as np
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


class Contrarec_Trainer():
    def __init__(self, args, user_num, item_num, cat_num, max_behaviors_length, max_items_length, max_cats_length, max_num_buy, train_negative_samples):
        self.cuda = args.cuda
        self.dataset = args.dataset
        self.dataset_path = './data/' + args.dataset + '/'
        self.user_num = user_num
        self.item_num = item_num
        self.cat_num = cat_num
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.hidden_dim = args.hidden_dim
        self.gat_layers_num = args.gat_layers_num
        self.heads_num = args.heads_num
        self.dropout = args.dropout
        self.epoch_num = args.epoch_num
        self.lr = args.lr
        self.wd = args.wd
        self.t = args.t
        self.beta = args.beta

        self.graph_encoder = args.graph_encoder
        self.fusion = args.fusion
        self.trigger = args.trigger

        self.max_behaviors_length = max_behaviors_length
        self.max_items_length = max_items_length
        self.max_cats_length = max_cats_length

        self.max_num_buy = max_num_buy
        self.train_negative_samples = train_negative_samples

        self.best_model = None

    def _load_batch_data(self, graphs_1, graphs_2, graphs_3, graphs_4):
        print('... Loading Batch Data for Trainer ...')
        nodes_num = 1 + self.max_behaviors_length + self.max_items_length + self.max_cats_length
        self.num_nodes = nodes_num
        for i in range(len(graphs_1)):
            graphs_1[i].num_nodes = nodes_num
            graphs_2[i].num_nodes = nodes_num
            graphs_3[i].num_nodes = nodes_num
            graphs_4[i].num_nodes = nodes_num

        total = list(zip(graphs_1, graphs_2, graphs_3, graphs_4))
        random.seed(self.seed)
        random.shuffle(total)
        graphs_1[:], graphs_2[:], graphs_3[:], graphs_4[:] = zip(*total)

        self.loader_1 = list(DataLoader(graphs_1, self.batch_size))
        self.loader_2 = list(DataLoader(graphs_2, self.batch_size))
        self.loader_3 = list(DataLoader(graphs_3, self.batch_size))
        self.loader_4 = list(DataLoader(graphs_4, self.batch_size))

        if self.cuda:
            print('... Convert Data to GPU ...')
            t = tqdm(range(len(self.loader_1)))
            for i in t:
                self.loader_1[i].to('cuda')
                self.loader_2[i].to('cuda')
                self.loader_3[i].to('cuda')
                self.loader_4[i].to('cuda')

    def _contrastive_loss(self, k_neg, k_pos, q):
        eps = 1e-8
        k_neg = k_neg.view(-1, self.hidden_dim, 1)
        k_pos = k_pos.view(-1, self.hidden_dim, 1)
        q = q.view(-1, 1, self.hidden_dim)

        l = -torch.log(torch.exp(torch.bmm(q, k_pos)/self.t)/(torch.exp(torch.bmm(q, k_pos)/self.t)+torch.exp(torch.bmm(q, k_neg)/self.t)+eps)+eps)
        l = torch.mean(l)

        return l

    def _rec_loss(self, scores, labels):
        eps = 1e-8
        mask = torch.where(labels!=-1, torch.ones_like(labels), torch.zeros_like(labels))

        l = -(torch.mul(labels, torch.log(scores+eps)) + torch.mul(torch.ones_like(labels)-labels, torch.log(torch.ones_like(scores)-scores+eps)))
        l = torch.mul(mask, l)
        l = torch.sum(l)/torch.sum(mask)

        return l

    def _loss(self, h_1_pos, h_2_neg, h_2_pos, h_3_neg, h_3_pos, h_4_neg, h_4_pos, scores, labels):
        l_c = self._contrastive_loss(h_1_pos, h_2_neg, h_2_pos) + self._contrastive_loss(h_2_pos, h_3_neg, h_3_pos) + self._contrastive_loss(h_3_pos, h_4_neg, h_4_pos)
        l_c = l_c/3

        l_rec = self._rec_loss(scores, labels)

        return l_c, l_rec

    def _save_train_log(self):
        self.log_folder_path = './log/' + self.dataset + '/HMG-CR_' + self.graph_encoder + '_' + self.fusion + '_h_' + str(self.hidden_dim) + '_g_layers_' + str(self.gat_layers_num) + '_heads_' + str(self.heads_num) + '_dropout_' + str(self.dropout) + '_lr_' + str(self.lr) + '_wd_' + str(self.wd) + '_t_' + str(self.t) + '_beta_' + str(self.beta) + '/'

        if not os.path.exists(self.log_folder_path):
            os.mkdir(self.log_folder_path)

        self.contrastive_loss_list = np.array(self.contrastive_loss_list).reshape(-1, 1)
        self.rec_loss_list = np.array(self.rec_loss_list).reshape(-1, 1)
        self.total_loss_list = np.array(self.total_loss_list).reshape(-1, 1)
        self.metric_list = np.array(self.metric_list).reshape(-1, 8)

        np.savetxt(self.log_folder_path+'contrastive_loss.csv', self.contrastive_loss_list, delimiter=',')
        np.savetxt(self.log_folder_path+'rec_loss.csv', self.rec_loss_list, delimiter=',')
        np.savetxt(self.log_folder_path+'total_loss.csv', self.total_loss_list, delimiter=',')
        np.savetxt(self.log_folder_path+'val_metrics.csv', self.metric_list, delimiter=',')

    def train(self):
        self.model = Contrarec_model(self.num_nodes, self.user_num, self.item_num, self.cat_num, self.hidden_dim, self.max_num_buy, self.train_negative_samples, self.graph_encoder, self.fusion, self.gat_layers_num, self.heads_num, self.dropout)
        if self.cuda:
            self.model.cuda()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        self.contrastive_loss_list = []
        self.rec_loss_list = []
        self.total_loss_list = []
        self.metric_list = []
        
        best_metric = 0
        trigger = self.trigger
        count = 0

        for epoch in range(self.epoch_num):
            self.model.mode = 'train'
            self.model.train()

            contrastive_loss = 0
            rec_loss = 0
            total_loss = 0
            metric = np.array([0 for j in range(8)])

            with trange(len(self.loader_1)) as t:
                for i in t:
                    t.set_description('Training Epoch %d Batch %d' % (epoch, i))
                    batch_data_1 = self.loader_1[i]
                    batch_data_2 = self.loader_2[i]
                    batch_data_3 = self.loader_3[i]
                    batch_data_4 = self.loader_4[i]

                    train_labels = batch_data_4.y.view(-1, 6, self.max_num_buy*(1+self.train_negative_samples))[:, 1, :].view(-1, self.max_num_buy*(1+self.train_negative_samples))
                    scores, h_1_pos, h_2_neg, h_2_pos, h_3_neg, h_3_pos, h_4_neg, h_4_pos = self.model(batch_data_1, batch_data_2, batch_data_3, batch_data_4, self.max_behaviors_length, self.max_items_length, self.max_cats_length)
                    #scores, d = self.model(batch_data_1, batch_data_2, batch_data_3, batch_data_4, self.max_behaviors_length, self.max_items_length, self.max_cats_length)
                    l_c, l_rec = self._loss(h_1_pos, h_2_neg, h_2_pos, h_3_neg, h_3_pos, h_4_neg, h_4_pos, scores, train_labels)
                    #l_c = torch.mean(d)
                    #l_rec = self._rec_loss(scores, train_labels)

                    t.set_postfix(contrastive_loss=l_c.item(), rec_loss=l_rec.item())
                    
                    if self.beta == 1:
                        l = 0.5*l_rec
                    else:
                        l = self.beta*l_rec + (1-self.beta)*l_c
                    contrastive_loss += l_c.item()
                    rec_loss += l_rec.item()
                    total_loss += l.item()

                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
            
            self.contrastive_loss_list.append(contrastive_loss/len(self.loader_1))
            self.rec_loss_list.append(rec_loss/len(self.loader_1))
            self.total_loss_list.append(total_loss/len(self.loader_1))

            ### Validation ###
            self.model.mode = 'val'
            self.model.eval()

            with torch.no_grad():
                with trange(len(self.loader_1)) as t:
                    for i in t:
                        t.set_description('Validation Epoch %d Batch %d' % (epoch, i))
                        batch_data_1 = self.loader_1[i]
                        batch_data_2 = self.loader_2[i]
                        batch_data_3 = self.loader_3[i]
                        batch_data_4 = self.loader_4[i]

                        val_labels = batch_data_4.y.view(-1, 6, self.max_num_buy*(1+self.train_negative_samples))[:, 3, :].view(-1, self.max_num_buy*(1+self.train_negative_samples))
                        scores, h_1_pos, h_2_neg, h_2_pos, h_3_neg, h_3_pos, h_4_neg, h_4_pos = self.model(batch_data_1, batch_data_2, batch_data_3, batch_data_4, self.max_behaviors_length, self.max_items_length, self.max_cats_length)
                        #scores, d = self.model(batch_data_1, batch_data_2, batch_data_3, batch_data_4, self.max_behaviors_length, self.max_items_length, self.max_cats_length)

                        val_labels = val_labels[:, :100]
                        scores = scores[:, :100]
                        evaluator = Evaluator(scores, val_labels)
                        metrics = evaluator.recalls_and_ndcgs_for_ks()

                        t.set_postfix(Recall_5=metrics['Recall@5'], NDCG_5=metrics['NDCG@5'])
                        
                        recalls = np.array([metrics['Recall@%d' % k] for k in [5, 10 ,20, 50]])
                        ndcgs = np.array([metrics['NDCG@%d' % k] for k in [5, 10, 20, 50]])
                        metric = metric + np.concatenate((recalls, ndcgs))

            metric = metric/len(self.loader_1)
            self.metric_list.append(metric)

            if metric[5] > best_metric:
                self.best_model = self.model
                best_metric = metric[5]
                print('The best model is at Epoch %d' % epoch)
                count = 0
            else:
                count += 1
                if count >= trigger:
                    break

        self._save_train_log()

    def test(self):
        self.best_model.mode = 'test'
        self.best_model.eval()
       
        metric = np.array([0 for j in range(8)])

        with torch.no_grad():
            with trange(len(self.loader_1)) as t:
                for i in t:
                    t.set_description('Testing Batch %d' % i)
                    batch_data_1 = self.loader_1[i]
                    batch_data_2 = self.loader_2[i]
                    batch_data_3 = self.loader_3[i]
                    batch_data_4 = self.loader_4[i]

                    test_labels = batch_data_4.y.view(-1, 6, self.max_num_buy*(1+self.train_negative_samples))[:, 5, :].view(-1, self.max_num_buy*(1+self.train_negative_samples))
                    scores, h_1_pos, h_2_neg, h_2_pos, h_3_neg, h_3_pos, h_4_neg, h_4_pos = self.best_model(batch_data_1, batch_data_2, batch_data_3, batch_data_4, self.max_behaviors_length, self.max_items_length, self.max_cats_length)
                    #scores, d = self.best_model(batch_data_1, batch_data_2, batch_data_3, batch_data_4, self.max_behaviors_length, self.max_items_length, self.max_cats_length)

                    test_labels = test_labels[:, :100]
                    scores = scores[:, :100]
                    evaluator = Evaluator(scores, test_labels)
                    metrics = evaluator.recalls_and_ndcgs_for_ks()
                    
                    t.set_postfix(Recall_5=metrics['Recall@5'], NDCG_5=metrics['NDCG@5'])

                    recalls = np.array([metrics['Recall@%d' % k] for k in [5, 10 ,20, 50]])
                    ndcgs = np.array([metrics['NDCG@%d' % k] for k in [5, 10, 20, 50]])
                    metric = metric + np.concatenate((recalls, ndcgs))

        metric = metric/len(self.loader_1)
        metric.reshape(1, -1)

        np.savetxt(self.log_folder_path+'test_result.csv', metric, delimiter=',')