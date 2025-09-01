import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class EmbedInit(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, is_norm=True):
        super(EmbedInit, self).__init__()
    
        self.gnn1 = GCNConv(input_dim, input_dim * 2)
        self.gnn2 = GCNConv(input_dim * 2, output_dim)
        self.is_norm = is_norm
        self.input_dim = input_dim

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(output_dim)

    def forward(self, embed, graph):
        graph_edge_index = graph.edge_index
        graph_x_embeddings = self.gnn1(embed, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        return graph_output