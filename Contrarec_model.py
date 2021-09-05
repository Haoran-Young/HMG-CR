import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, GCNConv, TAGConv, AGNNConv, GINConv, SGConv, APPNP
import time


class MLP(nn.Module):
    def __init__(self, layers, hidden_dim):
        super(MLP, self).__init__()

        self.mlp = nn.ModuleList([nn.Linear(layers[i]*hidden_dim, layers[i+1]*hidden_dim) for i in range(len(layers)-1)])
        self.activation = nn.ELU()

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

        return x


class Graph_encoder(nn.Module):
    def __init__(self, graph_encoder, gat_layers_num, hidden_dim, heads_num, dropout):
        super(Graph_encoder, self).__init__()

        if graph_encoder == 'GAT':
            self.gnn_layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim, heads_num, dropout=dropout, concat=False) for i in range(gat_layers_num)])
        elif graph_encoder == 'GCN':
            self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, add_self_loops=False) for i in range(gat_layers_num)])
        elif graph_encoder == 'TAG':
            self.gnn_layers = nn.ModuleList([TAGConv(hidden_dim, hidden_dim) for i in range(gat_layers_num)])
        elif graph_encoder == 'AGNN':
            self.gnn_layers = nn.ModuleList([AGNNConv() for i in range(gat_layers_num)])
        elif graph_encoder == 'GIN':
            self.gnn_layers = nn.ModuleList([GINConv(MLP([1, 2, 3, 1], hidden_dim), train_eps=True) for i in range(gat_layers_num)])
        elif graph_encoder == 'SG':
            self.gnn_layers = nn.ModuleList([SGConv(hidden_dim, hidden_dim, K=3, add_self_loops=False) for i in range(gat_layers_num)])
        elif graph_encoder == 'APPNP':
            self.gnn_layers = nn.ModuleList([APPNP(K=10, alpha=0.1, dropout=dropout) for i in range(gat_layers_num)])

        self.dropout = nn.Dropout(p=dropout)
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            x = self.elu(x)
            x = self.dropout(x)

        return x


class Contrarec_model(nn.Module):
    def __init__(self, num_nodes, users_num, items_num, cats_num, hidden_dim, max_num_buy, train_negative_samples, graph_encoder, fusion, gat_layers_num, heads_num, dropout):
        super(Contrarec_model, self).__init__()

        self.mode = 'train'
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_num_buy = max_num_buy
        self.train_negative_samples = train_negative_samples

        self.users_embeddings_table = nn.Embedding(users_num+1, hidden_dim, padding_idx=0)
        self.behaviors_embeddings_table = nn.Embedding(4+1, hidden_dim, padding_idx=0)
        self.items_embeddings_table = nn.Embedding(items_num+1, hidden_dim, padding_idx=0)
        self.cats_embeddings_table = nn.Embedding(cats_num+1, hidden_dim, padding_idx=0)
        # torch.nn.init.xavier_uniform_(self.users_embeddings_table.weight, gain=nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_uniform_(self.behaviors_embeddings_table.weight, gain=nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_uniform_(self.items_embeddings_table.weight, gain=nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_uniform_(self.cats_embeddings_table.weight, gain=nn.init.calculate_gain('relu'))

        self.graph_encoder_1 = Graph_encoder(graph_encoder, gat_layers_num, hidden_dim, heads_num, dropout)
        self.graph_encoder_2 = Graph_encoder(graph_encoder, gat_layers_num, hidden_dim, heads_num, dropout)
        self.graph_encoder_3 = Graph_encoder(graph_encoder, gat_layers_num, hidden_dim, heads_num, dropout)
        self.graph_encoder_4 = Graph_encoder(graph_encoder, gat_layers_num, hidden_dim, heads_num, dropout)

        # self.distance_mlp = MLP([2, 4, 2, 1], hidden_dim)
        # self.distance_mlp.activation = None
        # self.distance_mlp_2 = MLP([2, 4, 2, 1], hidden_dim)
        # self.distance_mlp_2.activation = None
        # self.distance_mlp_3 = MLP([2, 4, 2, 1], hidden_dim)
        # self.distance_mlp_3.activation = None

        # self.W_d = nn.Parameter(torch.Tensor(hidden_dim, 1))
        # self.W_d_2 = nn.Parameter(torch.Tensor(hidden_dim, 1))
        # self.W_d_3 = nn.Parameter(torch.Tensor(hidden_dim, 1))
        # torch.nn.init.xavier_normal_(self.W_d_1, gain=1)
        # torch.nn.init.xavier_normal_(self.W_d_2, gain=1)
        # torch.nn.init.xavier_normal_(self.W_d, gain=1)

        self.fusion = fusion

        # Concat Fusion
        if self.fusion == 'MLP':
            self.concat_mlp = MLP([4, 3, 2, 1], hidden_dim)

        # Personalized Non-linear Fusion
        if self.fusion == 'PNLF':
            self.W_u = nn.Embedding(users_num+1, 4, padding_idx=0)
            self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(4)])
            self.pnf_activation = nn.ELU()


        self.W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        torch.nn.init.xavier_normal_(self.W, gain=1)

    # def _distance(self, k_neg, k_pos, q, mlp, W_d):
    #     d_1_2 = mlp(torch.cat((k_neg, q), dim=1))
    #     d_1_2 = torch.sigmoid(torch.mm(d_1_2, W_d))

    #     d_2_2 = mlp(torch.cat((k_pos, q), dim=1))
    #     d_2_2 = torch.sigmoid(torch.mm(d_2_2, W_d))

    #     d_1_2_2 = -torch.log(torch.exp(d_2_2/0.01)/(torch.exp(d_1_2/0.01)+torch.exp(d_2_2/0.01)))

    #     return d_1_2_2

    def forward(self, batch_data_1, batch_data_2, batch_data_3, batch_data_4, max_behaviors_length, max_items_length, max_cats_length):
        edge_index_1 = batch_data_1.edge_index
        edge_index_2 = batch_data_2.edge_index
        edge_index_3 = batch_data_3.edge_index
        edge_index_4 = batch_data_4.edge_index

        x = batch_data_4.x.view(-1, self.num_nodes)
        y = batch_data_4.y.view(-1, 6, self.max_num_buy*(1+self.train_negative_samples))

        users_index = x[:, 0].view(-1, 1)
        behaviors_index = x[:, 1: max_behaviors_length+1]
        items_index = x[:, max_behaviors_length+1: max_items_length+max_behaviors_length+1]
        cats_index = x[:, -max_cats_length:]

        users_embeddings = self.users_embeddings_table(users_index)
        behaviors_embeddings = self.behaviors_embeddings_table(behaviors_index)
        items_embeddings = self.items_embeddings_table(items_index)
        cats_embeddings = self.cats_embeddings_table(cats_index)

        if self.mode == 'train':
            train_targets = y[:, 0, :].view(-1, self.max_num_buy*(1+self.train_negative_samples))
            targets_embeddings = self.items_embeddings_table(train_targets)
        elif self.mode == 'val':
            val_targets = y[:, 2, :].view(-1, self.max_num_buy*(1+self.train_negative_samples))
            targets_embeddings = self.items_embeddings_table(val_targets)
        elif self.mode == 'test':
            test_targets = y[:, 4, :].view(-1, self.max_num_buy*(1+self.train_negative_samples))
            targets_embeddings = self.items_embeddings_table(test_targets)

        x_embeddings = torch.cat([users_embeddings, behaviors_embeddings, items_embeddings, cats_embeddings], dim=1)
        x_embeddings = x_embeddings.view(-1, self.hidden_dim)

        h_1_pos = self.graph_encoder_1(x_embeddings, edge_index_1).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)

        h_2_neg = self.graph_encoder_1(x_embeddings, edge_index_2).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)
        h_2_pos = self.graph_encoder_2(x_embeddings, edge_index_2).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)

        h_3_neg = self.graph_encoder_2(x_embeddings, edge_index_3).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)
        h_3_pos = self.graph_encoder_3(x_embeddings, edge_index_3).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)

        h_4_neg = self.graph_encoder_3(x_embeddings, edge_index_4).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)
        h_4_pos = self.graph_encoder_4(x_embeddings, edge_index_4).view(-1, self.num_nodes, self.hidden_dim)[:, 0, :].view(-1, self.hidden_dim)

        ###### Distances Measurement ######
        # d_1_2_2 = self._distance(h_1_pos, h_2_neg, h_2_pos, self.distance_mlp, self.W_d)
        # d_2_3_3 = self._distance(h_2_pos, h_3_neg, h_3_pos, self.distance_mlp, self.W_d)
        # d_3_4_4 = self._distance(h_3_pos, h_4_neg, h_4_pos, self.distance_mlp, self.W_d)
        # d = (d_1_2_2 + d_2_3_3 + d_3_4_4)/3

        ###### Fusion Layer ######

        ### Concat ###
        if self.fusion == 'MLP':
            h_concat = torch.cat((h_1_pos, h_2_pos, h_3_pos, h_4_pos), dim=1)
            h_concat = self.concat_mlp(h_concat)
        ### Concat ###

        ### Personalized Non-linear Fusion ###
        if self.fusion == 'PNLF':
            user_weights = self.W_u(users_index).view(-1, 4)
            user_weights = torch.softmax(user_weights, dim=1)
            h_pos = [h_1_pos, h_2_pos, h_3_pos, h_4_pos]
            h_concat = torch.zeros_like(h_pos[0])
            for i in range(len(self.linears)):
                h = self.pnf_activation(self.linears[i](h_pos[i]))
                user_weight = user_weights[:, i].view(-1, 1).repeat(1, self.hidden_dim)
                h = torch.mul(user_weight, h)
                h_concat = h_concat + h
            h_concat = self.pnf_activation(h_concat)
        ### Personalized Non-linear Fusion ###

        ### SUM ###
        if self.fusion == 'SUM':
            h_concat = h_1_pos+h_2_pos+h_3_pos+h_4_pos
        ### SUM ###

        ### MEAN ###
        if self.fusion == 'MEAN':
            h_concat = (h_1_pos+h_2_pos+h_3_pos+h_4_pos)/4
        ### MEAN ###

        ###### Scores Prediction ######

        h_concat = h_concat.view(-1, self.hidden_dim, 1)
        latent_targets_embeddings = torch.matmul(targets_embeddings, self.W)
        scores = torch.bmm(latent_targets_embeddings, h_concat).view(-1, self.max_num_buy*(1+self.train_negative_samples))
        scores = torch.sigmoid(scores)

        #return scores, d
        return scores, h_1_pos, h_2_neg, h_2_pos, h_3_neg, h_3_pos, h_4_neg, h_4_pos