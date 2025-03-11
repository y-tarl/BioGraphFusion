# -*- coding:utf-8 -*-
# from torch.utils.tensorboard import SummaryWriter
import random
import os
import argparse

import torch
import numpy as np
from load_data import DataLoader_DisGeNet, DataLoader_STITCH, DataLoader_UMLS
from base_model import BaseModel
from utils import *
import torch, gc

gc.collect()
torch.cuda.empty_cache()


''' main script of BioGraphFuse'''
parser = argparse.ArgumentParser(description="Parser for BioGraphFuse")
parser.add_argument('--data_path', type=str, default='data/Disease-Gene/DisGeNet_cv')
parser.add_argument('--seed', type=int, default=
1234)
parser.add_argument('--gpu', type=int, default=7)
parser.add_argument('--topk', type=int, default=800)
parser.add_argument('--layers', type=int, default=6)
parser.add_argument('--sampling', type=str, default='incremental')
parser.add_argument('--weight', type=str, default=None)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--loss_in_each_layer', action='store_true')
parser.add_argument('--train', type=bool,default=True)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--HPO', action='store_true')
parser.add_argument('--eval_with_node_usage', action='store_true')
parser.add_argument('--scheduler', type=str, default='exp')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--fact_ratio', type=float, default=0.92)#0.9  剩下0.1用作真正的train
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--eval_interval', type=int, default=1)



parser.add_argument('--max_BKG_triples',type=int,default=15000)
optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer', choices=optimizers, default='Adam', help="Optimizer in {}".format(optimizers))
gate=['RNN', 'LSTM', 'GRU']
parser.add_argument('--gate',choices=gate,default='GRU',help='Gating mechanism in {}'.format(gate))
parser.add_argument('--rdim', default=32, type=int, help="Relation embedding dimensionality.")
parser.add_argument('--init', default=1e-3, type=float, help="Initial scale")
parser.add_argument('--lossflag', default=True, help='Whether to use N3 regularizer')
parser.add_argument('--Flag', default=True, help='Whether to use N3 regularizer')
parser.add_argument('--reg', default=0.1, type=float, help="Regularization weight")
parser.add_argument('--logFlag', default=True, help='Whether to write log')
parser.add_argument('--lamda', default=0.8569, type=float, help="scores weight")
args = parser.parse_args()

if args.data_path == 'data/Disease-Gene/DisGeNet_cv':
    args.BKG_list= ['disease-drug.txt', 'chemical-gene.txt']
elif args.data_path == 'data/Protein-Chemical/STITCH':
    args.BKG_list = ['disease-gene.txt', 'disease-drug.txt']



if __name__ == '__main__':
    opts = args
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    if dataset == 'DisGeNet_cv':
        DataLoader = DataLoader_DisGeNet
    elif dataset =='STITCH':
        DataLoader = DataLoader_STITCH
    elif dataset == 'UMLS':
        DataLoader = DataLoader_UMLS
        
    loader = DataLoader(opts)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel



    if  dataset=='DisGeNet_cv':
        opts.lr = 0.003
        opts.decay_rate = 0.994
        opts.lamb = 0.000017
        opts.hidden_dim =32
        opts.attn_dim = 5
        opts.dropout = 0.05
        opts.act = 'tanh'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 12

    elif dataset == 'STITCH':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim =64
        opts.attn_dim = 5
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10

    elif dataset == 'umls':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim =64
        opts.attn_dim = 5
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10




    # check all output paths
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{loader.task_dir}/saveModel/')


    model = BaseModel(opts, loader)


    opts.perf_file = f'results/{dataset}/{model.modelName}.txt'
    print(f'==> perf_file: {opts.perf_file}')

    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s,%d,%d,%d,%.4f,%.4f,%.4f\n' % (
    opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout,
    opts.act, opts.topk, opts.rdim, opts.seed, opts.reg,opts.lamda,opts.fact_ratio)
    print(config_str)
    if opts.logFlag:
        with open(opts.perf_file, 'a+') as f:
            f.write(config_str)

    if args.weight != None:
        model.loadModel(args.weight)
        model._update()
        model.model.updateTopkNums(opts.n_node_topk)
        print(model.model.lamda)

    if opts.train:
        # training model
        best_v_mrr = 0
        for epoch in range(opts.epoch):
            epoch_train_loss=model.train_batch()
            # eval on val/test set
            if (epoch+1) % args.eval_interval == 0:
                result_dict, out_str = model.evaluate(eval_val=True, eval_test=True)
                v_mrr, t_mrr = result_dict['v_mrr'], result_dict['t_mrr']
                print(out_str)
                if opts.logFlag:
                    with open(opts.perf_file, 'a+') as f:
                        f.write(out_str)
                if v_mrr > 0.1 and v_mrr > best_v_mrr:
                    best_v_mrr = v_mrr
                    best_str = out_str
                    print(str(epoch) + '\t' + best_str)
                    BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}_TestMRR_{str(t_mrr)[:5]}'#模型文件名更改处2
                    model.saveModelToFiles(BestMetricStr, deleteLastFile=False)
        gc.collect()
        torch.cuda.empty_cache()
        print(best_str)

    if opts.eval:
        # evaluate on test set with loaded weight to save time
        result_dict, out_str = model.evaluate(eval_val=False, eval_test=True, verbose=True)
        print(result_dict, '\n', out_str)

