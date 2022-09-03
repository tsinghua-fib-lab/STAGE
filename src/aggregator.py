import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.wd = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden_in, adj):
        # item transition
        h_locs_l, h_times_l, int_emb, h = hidden_in
        batch_size = h.shape[0]
        N = h.shape[1]

        h_ds = torch.matmul(torch.mean(torch.cat([h_locs_l, h_times_l, int_emb], dim=1), dim=1), self.wd).view(batch_size, 1, self.dim)  # (batchsize, 1, dim)


        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim) * h_ds + \
                  h.repeat(1, N, 1) * h_ds + \
                  h.repeat(1, 1, N).view(batch_size, N * N, self.dim) * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
        

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class HeteAggregator(nn.Module):
    # Aggregation with graph attentive convolution (GAC)
    def __init__(self, dim, dropout=0.0, alpha=0.2):
        super(HeteAggregator, self).__init__()
        self.dropout = dropout
        self.dim = dim

        self.w = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.q = nn.Parameter(torch.Tensor(self.dim, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, self_vectors, neighbor_vector):
        # self_vecs: (n, dim)
        # neighbor_vecs: (n, num_sample, dim)
        N = neighbor_vector.shape[1]
        h_node = self_vectors.unsqueeze(1).expand(-1, N, -1)
        
        neighbor_message = torch.matmul(neighbor_vector, self.w)
        attn_feat = h_node * neighbor_message
        attn_score = torch.matmul(self.leakyrelu(attn_feat), self.q).squeeze(-1)

        alpha = torch.softmax(attn_score, dim=-1).unsqueeze(-1)
        h_node_agg = torch.sum(alpha * neighbor_message, dim=-2)

        output = h_node_agg
        return output


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size, use_acf=True, bias=True):
        super(SemanticAttention, self).__init__()
        if use_acf:
            self.project = nn.Sequential(
                nn.Linear(in_size, hidden_size, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        else:
            self.project = nn.Sequential(
                nn.Linear(in_size, hidden_size, bias=bias),
                nn.Linear(hidden_size, 1, bias=False)
            )
    
    def attnw(self):
        return self.attnw
    
    def forward(self, z):
        """
        input z:(num_nodes, n_path, dim)
        """
        w = self.project(z)  # (num_nodes, n_path, 1)
        beta = torch.softmax(w, dim=1)
        self.attnw = beta
        return (beta * z).sum(1)  # (num_nodes, dim)