import datetime
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, HeteAggregator, SemanticAttention
from saframe import SAFrame
from torch.nn import Module, Parameter

 
class STAGE(SAFrame):
    def __init__(
        self, 
        opt, 
        num_locs,
        num_times,
        num_items,
        hete_graph,
        ):
        super().__init__(
            num_locs,
            num_times,
            num_items,
            embedding_dim=opt.hiddenSize,
            hete_graph=hete_graph,
            num_layers=int(opt.layer),
            feat_drop=0.0,
        )
        # HYPER PARA
        self.opt = opt 
        self.batch_size = opt.batch_size
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.n_factor = opt.n_factor
        self.sample_num = opt.n_sample
        self.hybrid = opt.hybrid
        self.layer = int(opt.layer)
        self.n_factor = opt.n_factor     # number of intention prototypes
        
        # Position representation & Intention prototypes
        self.pos_embedding = nn.Embedding(200, self.dim)
        self.intention_embedding = nn.Embedding(self.n_factor, self.dim)

        # Item Transition
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        # Generating Decision Factors
        self.l_agg = HeteAggregator(self.dim)
        self.t_agg = HeteAggregator(self.dim)
        self.int_agg = []
        for _ in range(self.n_factor):
            self.int_agg.append(HeteAggregator(self.dim))
        # Generating User Characteristics
        self.s_agg = HeteAggregator(self.dim)

        # intent-attn
        self.int_attn = SemanticAttention(in_size=self.dim, hidden_size=self.dim, use_acf=True, bias=True)

        self.w_p = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_3 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_4 = nn.Parameter(torch.Tensor(self.dim, self.dim))

        self.q_attn = nn.Parameter(torch.Tensor(size=(self.dim, 1))) 
        self.linear_transform = nn.Linear(self.dim * 2, self.dim, bias=False)
        
        
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, h_locs, h_times, int_embs, h_s):
        
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]

        
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # add pos_emb
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_p) # add pos_emb
        nh = torch.tanh(nh)

        h_int = self.int_attn(torch.stack(int_embs, dim=0).permute(1,0,2)) #(b, dim)
        

        ht = torch.matmul(h_int, self.w_1) + torch.matmul(h_locs, self.w_2) + torch.matmul(h_times, self.w_3) + torch.matmul(h_s, self.w_4)
        ht = ht.unsqueeze(1).repeat(1, len, 1)             # (b, N, dim)
        
        feat = ht * nh  
        feat_s = torch.sigmoid(feat)  
        attn_s = feat_s.matmul(self.q_attn)  # (b, N, 1)
        beta = attn_s * mask

        b = self.item_embedding.weight[1:]  # n_nodes x latent_size

        select = torch.sum(beta * hidden, 1)
        if self.hybrid:
            select = self.linear_transform(torch.cat([select, hidden[:, 0, :]], dim=-1))
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def trans_to_seq(self, alias_inputs, hidden):
        # map session graph to seq
        get = lambda index: hidden[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return seq_hidden

    def forward(self, alias_inputs, inputs, Hs, mask_item, item, locs, times):
        # item: item seq with 0 as mask. 
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
       
        h = self.item_embedding(inputs)
        item_emb = self.item_embedding(item) * mask_item.float().unsqueeze(-1)

        session_c = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        session_c = session_c.unsqueeze(1)  # (batchsize, 1, dim)
        h_locs_l = self.loc_embedding(locs).unsqueeze(1)     # (batchsize, 1, dim)
        h_times_l = self.time_embedding(times).unsqueeze(1)  # (batchsize, 1, dim)
        int_emb = self.intention_embedding.weight # (k, dim)
        int_emb = int_emb.unsqueeze(0).repeat(batch_size, 1, 1)      # (b , k, dim)
        
        h_local_in = [h_locs_l, h_times_l, int_emb, h]
        
        # local
        # item transition
        h_local = self.local_agg(h_local_in, Hs) # items: (batchsize, seq, dim)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)

        # Generating Decision Factors
        h_locs_l = self.l_agg(h_locs_l.squeeze(), h_local) + h_locs_l.squeeze()
        h_times_l = self.t_agg(h_times_l.squeeze(), h_local) + h_times_l.squeeze()
        int_embs = []
        for int_id in range(self.n_factor):
            int_embs.append(self.int_agg[int_id](self.intention_embedding(trans_to_cuda(torch.tensor(int_id)).long()).unsqueeze(0).repeat(batch_size, 1) , h_local))  # (b, dim)

        # Generating User Characteristics
        h_ds = torch.stack([h_locs_l, h_times_l] + int_embs, dim=0) #(k+2, b, dim)
        h_s = self.s_agg(session_c.squeeze(), h_ds.permute(1,0,2)) + session_c.squeeze()

        h_local_seq = self.trans_to_seq(alias_inputs, h_local)

        # global
        h_global_seq, h_locs_g, h_times_g = super().forward(item, locs, times, session_c)
        h_global_seq = h_global_seq.view(batch_size, seqs_len, self.dim)

        h_seq = h_local_seq + h_global_seq
        h_locs = h_locs_g + h_locs_l
        h_times = h_times_g + h_times_l

        return h_seq, h_locs, h_times, int_embs, h_s


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, Hs, items, mask, targets, inputs, locs, times = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    Hs = trans_to_cuda(Hs).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    locs = trans_to_cuda(locs).long()
    times = trans_to_cuda(times).long()

    seq_hidden, h_locs, h_times, int_embs, h_s = model(alias_inputs, items, Hs, mask, inputs, locs, times)
    return targets, model.compute_scores(seq_hidden, mask, h_locs, h_times, int_embs, h_s)


def train_test(model, train_data, test_data, top_K):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)

    for data in test_loader:
        targets, scores = forward(model, data)
        targets = targets.numpy()
        for K in top_K:
            sub_scores = scores.topk(K)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                metrics['hit%d' % K].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    return metrics
