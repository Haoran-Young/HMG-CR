import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import numpy as np
from DataLoader import DataLoader
from BatchLoader_Contrarec import BL_Contrarec
from BatchLoader_NMTR import BL_NMTR
from BatchLoader_Graph import BL_Graph
from Contrarec_Trainer import Contrarec_Trainer
from NMTR_Trainer import NMTR_Trainer
from Graph_Trainer import Graph_Trainer
import torch

def main():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--model_name', action='store', type=str, default='contrarec')

    ###### DataLoader Parameters ######
    my_parser.add_argument('--cuda', action='store_true', default=False)
    my_parser.add_argument('--dataset', action='store', type=str, default='taobao')
    my_parser.add_argument('--min_num_buy', action='store', type=int, default=5)
    my_parser.add_argument('--max_num_buy', action='store', type=int, default=100)
    my_parser.add_argument('--seed', action='store', type=int, default=12345)
    my_parser.add_argument('--train_negative_samples', action='store', type=int, default=5)
    my_parser.add_argument('--batch_size', action='store', type=int, default=64)

    my_parser.add_argument('--graph_encoder', action='store', type=str, default='GCN')
    my_parser.add_argument('--fusion', action='store', type=str, default='MLP')
    my_parser.add_argument('--trigger', action='store', type=int, default=30)

    my_parser.add_argument('--hidden_dim', action='store', type=int, default=16)
    my_parser.add_argument('--gat_layers_num', action='store', type=int, default=3)
    my_parser.add_argument('--dropout', action='store', type=float, default=0.2)
    my_parser.add_argument('--heads_num', action='store', type=int, default=3)
    my_parser.add_argument('--epoch_num', action='store', type=int, default=50)
    my_parser.add_argument('--lr', action='store', type=float, default=1e-4)
    my_parser.add_argument('--wd', action='store', type=float, default=1e-6)
    my_parser.add_argument('--t', action='store', type=float, default=0.01)
    my_parser.add_argument('--beta', action='store', type=float, default=0.5)

    args = my_parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###### Data Loading ######
    dataloader = DataLoader(args.dataset, args.min_num_buy, args.train_negative_samples)
    #dataloader._get_statistics_file()
    dataloader.load()
    dataloader.train_validation_test_data_generator()

    ###### Batch Data Loading for Models ######
    if args.model_name == 'contrarec':
        batchloader = BL_Contrarec(args.dataset, args.max_num_buy, args.train_negative_samples, args.batch_size)
        batchloader.load()
        nodes_num = batchloader.max_behaviors_length + batchloader.max_items_length + batchloader.max_cats_length + 1
    
    if args.model_name == 'contrarec':
        trainer = Contrarec_Trainer(args, batchloader.user_num, batchloader.item_num, batchloader.cat_num, batchloader.max_behaviors_length, batchloader.max_items_length, batchloader.max_cats_length, args.max_num_buy, args.train_negative_samples)
        trainer._load_batch_data(batchloader.graphs_1, batchloader.graphs_2, batchloader.graphs_3, batchloader.graphs_4)
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()