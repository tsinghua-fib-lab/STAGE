import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from aggregator import HeteAggregator, SemanticAttention

def trans_Hg_to_cuda(Hg):
    # Hg:dict, Hg['edge_type']['neighbors']    Hg['edge_type']['weight']
    edge_types = ['LT', 'TL', 'LI', 'IL', 'TI', 'IT', 'II']
    if torch.cuda.is_available():
        for edge_type in edge_types:
            Hg[edge_type]['neighbor'] = torch.Tensor(np.array(Hg[edge_type]['neighbor'])).cuda().long()
            Hg[edge_type]['weight'] = torch.Tensor(np.array(Hg[edge_type]['weight'])).cuda().float()
    else:
        print("GPU is unavailable, using CPU!")
        for edge_type in edge_types:
            Hg[edge_type]['neighbor'] = torch.Tensor(np.array(Hg[edge_type]['neighbor'])).cpu().long()
            Hg[edge_type]['weight'] = torch.Tensor(np.array(Hg[edge_type]['weight'])).cpu().float()
    
    return Hg


# Scene-aware Frame for Session-based Recommendation (GraphSAGE)
class SAFrame(nn.Module):
    def __init__(
        self,
        num_locs,
        num_times,
        num_items,
        embedding_dim,
        hete_graph,
        num_layers=1,
        feat_drop=0.0,
    ):
        super().__init__()
        self.hop = num_layers  # GraphSAGE: n_hop = num_layers (1 layer)
        self.loc_embedding = nn.Embedding(num_locs, embedding_dim, padding_idx=0)
        self.time_embedding = nn.Embedding(num_times, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        self.loc_indices = torch.arange(num_locs, dtype=torch.long)
        self.time_indices = torch.arange(num_times, dtype=torch.long)
        self.item_indices = torch.arange(num_items, dtype=torch.long)
        
        self.Hg = trans_Hg_to_cuda(hete_graph)
        # Edge Type: {e=uv|u->v}   [LT,TL,LI,IL,TI,IT,II]
        # item level
        self.LT_agg = HeteAggregator(embedding_dim, feat_drop)
        self.TL_agg = HeteAggregator(embedding_dim, feat_drop)
        self.LI_agg = HeteAggregator(embedding_dim, feat_drop)
        self.IL_agg = HeteAggregator(embedding_dim, feat_drop)
        self.TI_agg = HeteAggregator(embedding_dim, feat_drop)
        self.IT_agg = HeteAggregator(embedding_dim, feat_drop)
        self.II_agg = HeteAggregator(embedding_dim, feat_drop)
        # session level: session->L, T
        # self.SL_agg = HeteAggregator(embedding_dim, feat_drop)
        # self.ST_agg = HeteAggregator(embedding_dim, feat_drop)

        # inter-meta-relation agg
        self.inter_agg_l = SemanticAttention(in_size=embedding_dim, hidden_size=embedding_dim, use_acf=True, bias=True)
        self.inter_agg_t = SemanticAttention(in_size=embedding_dim, hidden_size=embedding_dim, use_acf=True, bias=True)
        self.inter_agg_i = SemanticAttention(in_size=embedding_dim, hidden_size=embedding_dim, use_acf=True, bias=True)
        

    def forward(self, item, locs, times, session_emb):
        # items:(batchsize, seq_len), locs:(batchsize,)  times:(batchsize, )
        batch_size = item.shape[0]
        seqs_len = item.shape[1]
        iids = item.view(-1)
        lids = locs
        tids = times

        # intra-relation aggregation
        # item level propagation
        # sample neighbors in hete_graph
        l_ni = self.Hg['IL']['neighbor'][lids]  # location's item neighbor id
        l_nt = self.Hg['TL']['neighbor'][lids]  # location's time neighbor
        t_ni = self.Hg['IT']['neighbor'][tids]  # time's item neighbor
        t_nl = self.Hg['LT']['neighbor'][tids]  # time's location neighbor
        i_ni = self.Hg['II']['neighbor'][iids]  # item's item neighbor
        i_nl = self.Hg['LI']['neighbor'][iids]  # item's location neighbor
        i_nt = self.Hg['TI']['neighbor'][iids]  # item's time neighbor

        h_l_IL = self.IL_agg(self.loc_embedding(lids), self.item_embedding(l_ni))
        h_l_TL = self.TL_agg(self.loc_embedding(lids), self.time_embedding(l_nt))
        h_t_IT = self.IT_agg(self.time_embedding(tids), self.item_embedding(t_ni))
        h_t_LT = self.LT_agg(self.time_embedding(tids), self.loc_embedding(t_nl))
        h_i_II = self.II_agg(self.item_embedding(iids), self.item_embedding(i_ni))
        h_i_LI = self.LI_agg(self.item_embedding(iids), self.loc_embedding(i_nl))
        h_i_TI = self.TI_agg(self.item_embedding(iids), self.time_embedding(i_nt))

        # session level propagation in a batch to reduce computing cost
        # in a mini-batch, each session only belongs to a certain location, time, 
        # we use the session embedding to enhance the modeling of location and time.
        h_l_SL = session_emb.squeeze(1)
        h_t_ST = session_emb.squeeze(1)

        # inter-relation aggregation
        h_l_stack = torch.stack([h_l_IL, h_l_TL, h_l_SL], dim=1)
        h_t_stack = torch.stack([h_t_IT, h_t_LT, h_t_ST], dim=1)
        h_i_stack = torch.stack([h_i_II, h_i_TI, h_i_LI], dim=1)


        h_items = self.inter_agg_i(h_i_stack)
        h_locs = self.inter_agg_l(h_l_stack)
        h_times = self.inter_agg_t(h_t_stack)

        return h_items, h_locs, h_times