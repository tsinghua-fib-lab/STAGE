import time
import argparse
import pickle
import os
import numpy as np
from sessionG import *
from utils import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='Beijing/Shanghai')
parser.add_argument('--model', default='STAGE', help='[STAGE]')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--n_factor', type=int, default=5, help='Disentangle factors number')
parser.add_argument('--hybrid', action='store_true', help='user hybrid')
parser.add_argument('--gpu_id', type=str,default="1")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--n_iter', type=int, default=1)                                    
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')    
parser.add_argument('--dropout_global', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--g', action='store_true', help='use g')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    init_seed(2020)

    if opt.dataset == 'Beijing':
        num_locs = 14
        num_times = 97
        num_items = 42902
        opt.n_iter = 1
        opt.dropout_gcn = 0.1
        opt.dropout_local = 0.0
    elif opt.dataset == 'Shanghai':
        num_locs = 14
        num_times = 97
        num_items = 37682
        opt.n_iter = 1
        opt.dropout_gcn = 0.1
        opt.dropout_local = 0.0
    else:
        num_locs = 12
        num_times = 94
        num_items = 2307
        opt.n_iter = 1
        opt.dropout_gcn = 0.1
        opt.dropout_local = 0.0

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))



    train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    Hg = build_hete_graph(train_seq, num_locs, num_times, num_items, sample_num=12) # dict
    train_data = Data(train_data, num_int=opt.n_factor)
    test_data = Data(test_data, num_int=opt.n_factor)

    
    model = trans_to_cuda(STAGE(opt, num_locs, num_times, num_items, Hg))

    print(opt)
    start = time.time()

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics = train_test(model, train_data, test_data, top_K)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                flag = 1
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                flag = 1
        for K in top_K:
            print('Current Result:')
            print('\tP@%d: %.4f\tMRR%d: %.4f' %
                (K, metrics['hit%d' % K], K, metrics['mrr%d' % K]))
            print('Best Result:')
            print('\tP@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
