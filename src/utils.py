import math
import random
import itertools
import numpy as np
import torch
import pickle
import scipy.sparse as sp 
from tqdm import tqdm
from torch.utils.data import Dataset

def build_hete_graph(train_seq, num_locs, num_times, num_items, sample_num=12):
    # Node type: [location, time, item]    {(i, l, t)} min_id=1
    # Edge Type: {e=uv|u->v}   [LT,TL,LI,IL,TI,IT,II] item-level edges
    # sampled Hg:dict, Hg['edge_type']['neighbors']    Hg['edge_type']['weight']
    print('building heterogeneous graph...')
    node_types = ['location', 'time', 'item']
    edge_types = ['LT', 'TL', 'LI', 'IL', 'TI', 'IT', 'II']
    Hg = dict()
    Hg['node_types'] = node_types
    Hg['edge_types'] = edge_types
    for edge_type in edge_types:
        Hg[edge_type] = dict()
        src, des = edge_type[0], edge_type[1]
        if des == 'L':
            num = num_locs
        elif des == 'T':
            num = num_times
        else:
            num = num_items
        relation = []
        adj1 = [dict() for _ in range(num)]
        adj = [[] for _ in range(num)]
        for i in range(len(train_seq)):
            data = train_seq[i]
            if edge_type == 'II':
                for k in range(1, 4):
                    for j in range(len(data)-k):
                        relation.append([data[j][0], data[j+k][0]])
                        relation.append([data[j+k][0], data[j][0]])
                        '''
                        src_node = data[i][0]
                        des_node = data[i+1][0]
                        relation.append([src_node, des_node])
                        '''
                    
            else:
                for (i, l, t) in data:
                    if src == 'L':
                        src_node = l
                    elif src == 'T':
                        src_node = t
                    else:
                        src_node = i

                    if des == 'L':
                        des_node = l
                    elif des == 'T':
                        des_node = t
                    else:
                        des_node = i
                    relation.append([src_node, des_node])
        
        # tuple: (src_node, des_node)
        for tup in relation:
            if tup[0] in adj1[tup[1]].keys():
                adj1[tup[1]][tup[0]] += 1
            else:
                adj1[tup[1]][tup[0]] = 1

        weight = [[] for _ in range(num)]

        for t in range(num):
            x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]   # (src, value)
            adj[t] = [v[0] for v in x]
            weight[t] = [v[1] for v in x]
        # sampe_all = 2*sample_num
        sample_all = 2*sample_num
        for i in range(num):
            adj[i] = adj[i][:sample_all]
            weight[i] = weight[i][:sample_all]
        # sample
        neighbor_sample, weight_sample = handle_adj(adj, num, sample_num, weight)
        Hg[edge_type]['neighbor'] = neighbor_sample
        Hg[edge_type]['weight'] = weight_sample

    return Hg
        


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

    
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData):
    items, locs, times, len_data = [], [], [], []
    for nowData in inputData:
        len_data.append(len(nowData))
        Is = []
        for (i, l, t) in nowData:
            Is.append(i)
        locs.append(l)
        times.append(t)
        items.append(Is)
    # len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)

    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(items, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]

    return us_pois, us_msks, max_len, locs, times



class Data(Dataset):
    def __init__(self, data, num_int=5):
        inputs, mask, max_len, locs, times = handle_data(data[0])
        self.inputs = np.asarray(inputs)
        self.locs = locs
        self.times = times
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len # max_node_num
        self.num_int = num_int

    def __getitem__(self, index):
        u_input, mask, u_loc, u_time, target = self.inputs[index], self.mask[index], self.locs[index], self.times[index], self.targets[index]

        node = np.unique(u_input)  # zero is mask
        items = node.tolist() + (self.max_len - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]     

        max_n_node = self.max_len # item_num
        # Construct Session Graph
        adj = np.zeros((max_n_node, max_n_node)) # items, 0
        
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        
        # alias_input
        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), 
                torch.tensor(u_loc), torch.tensor(u_time)]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    train_data = pickle.load(open('../datasets/sample/train.txt', 'rb'))
    inputs, mask, max_len, max_edge_num, locs, times = handle_data(train_data[0], sw=[2, 3])
    print(max_len, max_edge_num)
    node = np.unique(inputs[0])  # zero is mask
    
    
    train_data = Data(train_data, num_int=5)
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=100,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        alias_inputs, Hs, items, mask, targets, inputs, locs, times = data
    
        
    
    