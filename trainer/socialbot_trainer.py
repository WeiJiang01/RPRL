from utils.tools import accuracy
from utils.logger import (
    get_logger, 
    get_log_dir,
)
import time
import os
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc
import torch
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle
import random
from collections import deque, defaultdict


class SocialbotTrainer(object):
    def __init__(self, dataset, model, optimizer, loss, args, device):
        super(SocialbotTrainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.args = args

        self.x_data, self.edge_index, self.edge_type, self.labels, self.train_idx, self.val_idx, self.test_idx = dataset

        
        print('Loading neighbors...')
        self.neighbor_matrix, self.hop_label, self.edge_index_list = self.get_all_neighbors_matrix(self.x_data, self.edge_index, args.neighbor_num, n_hops=args.n_hops)
        print('Finished.')

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

  
    def get_all_neighbors_matrix(self, x_data, edge_index, neighbor_num, n_hops):
        device = x_data.device
        num_nodes, hidden_dim = x_data.size()
        total_nb = neighbor_num * n_hops

        # build adjacency list once
        adj = [[] for _ in range(num_nodes)]
        for u, v in edge_index.t().tolist():
            adj[u].append(v)
            adj[v].append(u)

        # pre-alloc outputs
        x_data_n   = torch.zeros(num_nodes, total_nb, hidden_dim, device=device)
        hop_labels = torch.zeros(num_nodes, total_nb, dtype=torch.long, device=device)
        edge_index_list = []
        global_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)

        for node in tqdm(range(num_nodes)):
            # BFS + sampling
            sampled, labels = [], []
            visited = {node}
            frontier = deque([node])
            for hop in range(1, n_hops+1):
                next_front = set()
                while frontier:
                    cur = frontier.popleft()
                    for w in adj[cur]:
                        if w not in visited:
                            visited.add(w)
                            next_front.add(w)
                frontier = deque(next_front)

                layer = list(next_front)
                if len(layer) >= neighbor_num:
                    picks = random.sample(layer, neighbor_num)
                elif layer:
                    picks = random.choices(layer, k=neighbor_num)
                else:
                    picks = [node]*neighbor_num

                sampled.extend(picks)
                labels.extend([hop]*neighbor_num)

            x_data_n[node]   = x_data[torch.tensor(sampled, device=device)]
            hop_labels[node] = torch.tensor(labels, device=device)

            # reindex edges
            global_map.fill_(-1)
            global_map[torch.tensor(sampled, device=device)] = torch.arange(total_nb, device=device)
            mask = (global_map[edge_index[0]] >= 0) & (global_map[edge_index[1]] >= 0)
            sub = edge_index[:, mask]
            edge_index_list.append(global_map[sub])

        return x_data_n, hop_labels, edge_index_list


    def train_process(self):

        start_time = time.time()

        # self.model.apply(init_weights)

        for epoch in range(self.args.n_epochs):

            self.model.train()
            self.optimizer.zero_grad()

            data = [self.x_data, self.edge_index, self.neighbor_matrix, self.edge_type, self.hop_label, self.edge_index_list]

            output, state_probs, beta, edge_index_list = self.model(data)

            train_loss = self.loss(output[self.train_idx], self.labels[self.train_idx])
            train_edge_index_list = [edge_index_list[i] for i in self.train_idx.tolist()]
            train_state_probs = state_probs[self.train_idx]

            pde_loss = self.model.pde_loss(train_state_probs, beta, train_edge_index_list)
    
            train_loss += self.args.alpha*torch.log(1+pde_loss)
            train_loss.backward()
            self.optimizer.step()

            self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_loss))

            # acc_train = accuracy(output[train_idx], labels[train_idx])

            self.model.eval()
            with torch.no_grad():

                val_loss = self.loss(output[self.val_idx], self.labels[self.val_idx])
                
                self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
            
            if epoch == 39 or epoch == 49 or epoch == 59 or epoch == 69 or epoch == 79 or epoch == 89:
                self.test()

        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    .format(
                        (training_time / 60), 
                        ))
        
        if self.args.train_mode == 'pretrain':
            torch.save(self.model.state_dict(), "pretrained_models/"+self.args.pretrain_dataset+"_rprl_pretrained.pth")

        acc, auc, f1 = self.test()

        return acc, auc, f1

    def test(self):
        self.model.eval()
        with torch.no_grad():
            data = [self.x_data, self.edge_index, self.neighbor_matrix, self.edge_type, self.hop_label, self.edge_index_list]
            output, _,_,_ = self.model(data)
            acc_test = accuracy(output[self.test_idx], self.labels[self.test_idx])
            output = output.max(1)[1].to('cpu').detach().numpy()
            label = self.labels.to('cpu').detach().numpy()
            f1=f1_score(label[self.test_idx],output[self.test_idx])
            precision=precision_score(label[self.test_idx],output[self.test_idx])
            recall=recall_score(label[self.test_idx],output[self.test_idx])
            fpr, tpr, thresholds = roc_curve(label[self.test_idx], output[self.test_idx], pos_label=1)
            Auc=auc(fpr, tpr)
            
            self.logger.info("Test Acc: {:.4f} | Prec: {:.4f} | Rec: {:.4f} | F1: {:.4f} | AUC {:.4f}".format(acc_test.item(), precision.item(), recall.item(), f1.item(), Auc.item()))

            return acc_test.item(), Auc.item(), f1.item()
