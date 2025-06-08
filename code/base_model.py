# -*- coding:utf-8 -*-
import torch
import numpy as np
import time
import os
from torch.optim import SparseAdam
from torch.optim import Adam
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from models import GNNModel
from utils import *
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Initialize the gradient scaler
scaler = GradScaler()


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = GNNModel(args, loader)
        self.model.cuda()
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.args = args
        if args.optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        else :
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)

        if self.args.scheduler == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        else:
            raise NotImplementedError(f'==> [Error] {self.scheduler} scheduler is not supported yet.')
        
        self.t_time = 0
        self.lastSaveGNNPath = None
        self.modelName = f'{args.n_layer}-layers'
        # for i in range(args.n_layer):
        #     i_n_node_topk = args.n_node_topk if 'int' in str(type(args.n_node_topk)) else args.n_node_topk[i]
        #     self.modelName += f'-{i_n_node_topk}'
        self.modelName += f'-{args.topk}'
        if hasattr(loader, 'BKG_list'):
            for i in loader.BKG_list:
                self.modelName += f'-{i}'
            self.modelName += f'-{args.max_BKG_triples}'
        print(f'==> model name: {self.modelName}')

    def _update(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
        
    def saveModelToFiles(self, best_metric, deleteLastFile=True):
        savePath = f'{self.loader.task_dir}/saveModel/{self.modelName}-{best_metric}.pt'
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mrr':best_metric,
                }, savePath)

        if deleteLastFile and self.lastSaveGNNPath != None:
            print(f'Remove last checkpoint: {self.lastSaveGNNPath}')
            os.remove(self.lastSaveGNNPath)

        self.lastSaveGNNPath = savePath

    def loadModel(self, filePath, layers=-1):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))
        if layers != -1:
            extra_layers = self.model.gnn_layers[layers:]
            self.model.gnn_layers = self.model.gnn_layers[:layers]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.gnn_layers += extra_layers
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print(f"Failed to load optimizer state: {e}. Skipping optimizer load.")

    def train_batch(self,):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        t_time = time.time()
        self.model.train()
        
        for i in tqdm(range(n_batch), position=0):
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()


            scores,l = self.model(triple[:,0], triple[:,1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:,2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) + l


            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():#
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        self.t_time += time.time() - t_time

        if self.args.scheduler == 'exp':
            self.scheduler.step()

        self.loader.shuffle_train()

        return epoch_loss

    def evaluate(self, verbose=True, eval_val=True, eval_test=False, inference_path=False,writer_flag=False):
        batch_size = self.n_tbatch
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        with (torch.no_grad()):
            # - - - - - - val set - - - - - -
            if not eval_val:
                v_mrr, v_h1, v_h3, v_h10, v_h50, v_map_1, v_map_3, v_map_10, v_map_50 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            else:
                iterator = tqdm(range(n_batch), position=0) if verbose else range(n_batch)
                num=[]
                for i in iterator:
                    start = i*batch_size
                    end = min(n_data, (i+1)*batch_size)
                    batch_idx = np.arange(start, end)
                    subs, rels, objs, nums = self.loader.get_batch(batch_idx, data='valid')
                    num+=nums
                    scores = self.model(subs, rels, mode='valid', inference_path=inference_path)
                    scores = scores.data.cpu().numpy()

                    filters = []

                    for i in range(len(subs)):
                        filt = self.loader.filters[(subs[i], rels[i])]
                        filt_1hot = np.zeros((self.n_ent, ))
                        filt_1hot[np.array(filt)] = 1
                        filters.append(filt_1hot)

                    # scores / objs / filters: [batch_size, n_ent]
                    filters = np.array(filters)
                    ranks = cal_ranks(scores, objs, filters)
                    ranking += ranks

                ranking = np.array(ranking)
                v_mrr, v_h1, v_h3 ,v_h10,v_h50,v_map_1, v_map_3, v_map_10, v_map_50 = cal_performance(ranking,num)

            # - - - - - - test set - - - - - -
            if not eval_test:
                t_mrr, t_h1, t_h10 = -1, -1, -1
            else:
                n_data = self.n_test
                n_batch = n_data // batch_size + (n_data % batch_size > 0)
                ranking = []
                self.model.eval()
                iterator = tqdm(range(n_batch), position=0) if verbose else range(n_batch)
                num=[]
                for i in iterator:
                    start = i*batch_size
                    end = min(n_data, (i+1)*batch_size)
                    batch_idx = np.arange(start, end)
                    subs, rels, objs,nums = self.loader.get_batch(batch_idx, data='test')
                    num += nums
                    scores = self.model(subs, rels, mode='test', inference_path=inference_path)
                    scores = scores.data.cpu().numpy()
                    filters = []
                    for i in range(len(subs)):
                        filt = self.loader.filters[(subs[i], rels[i])]
                        filt_1hot = np.zeros((self.n_ent, ))
                        filt_1hot[np.array(filt)] = 1
                        filters.append(filt_1hot)

                    filters = np.array(filters)
                    ranks = cal_ranks(scores, objs, filters)
                    ranking += ranks

                ranking = np.array(ranking)
                t_mrr, t_h1, t_h3 ,t_h10,t_h50,t_map_1, t_map_3, t_map_10, t_map_50 = cal_performance(ranking,num)

            i_time = time.time() - i_time
            out_str = (
                    '[VALID] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\t H@50:%.4f MAP@1:%.4f MAP@3:%.4f MAP@10:%.4f MAP@50:%.4f\t'
                    '[TEST] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\t H@50:%.4f MAP@1:%.4f MAP@3:%.4f MAP@10:%.4f MAP@50:%.4f\t'
                    '[TIME] train:%.4f inference:%.4f\n'
                    % (
                        v_mrr, v_h1, v_h3 , v_h10, v_h50, v_map_1, v_map_3, v_map_10, v_map_50,
                        t_mrr, t_h1, t_h3, t_h10, t_h50, t_map_1, t_map_3, t_map_10, t_map_50,
                        self.t_time, i_time
                    )
            )

            result_dict = {
                'v_mrr': v_mrr,
                'v_h1': v_h1,
                'v_h3': v_h3,
                'v_h10': v_h10,
                'v_h50': v_h50,
                'v_map_1': v_map_1,
                'v_map_3': v_map_3,
                'v_map_10': v_map_10,
                'v_map_50': v_map_50,
                't_mrr': t_mrr,
                't_h1': t_h1,
                't_h3': t_h3,
                't_h10': t_h10,
                't_h50': t_h50,
                't_map_1': t_map_1,
                't_map_3': t_map_3,
                't_map_10': t_map_10,
                't_map_50': t_map_50
            }

        return result_dict, out_str