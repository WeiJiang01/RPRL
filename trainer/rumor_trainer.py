import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from dataloaders.rumor_dataloader import *
import time
import os
from utils.word2vec import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from utils.logger import (
    get_logger, 
    get_log_dir,
)


class RumorTrainer(object):
    def __init__(self, datasets, model, optimizer, args, device):
        super(RumorTrainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.device = device
        self.args = args

        train_dataset, val_dataset, test_dataset = datasets

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        self.train_per_epoch = len(self.train_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)        
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))
        self.logger.info('Training Mode: {}'.format(args.train_mode))
    
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            data.to(self.device)
            out_labels, state_probs, beta, edge_index_list = self.model(data)
            loss = F.nll_loss(out_labels, data.y)
            p_loss = self.model.pde_loss(state_probs, beta, edge_index_list)
            loss = loss + self.args.alpha*torch.log(1+p_loss)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_epoch_loss = train_loss/self.train_per_epoch

        self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss

    def validate_epoch(self, epoch):

        val_losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data.to(self.device)
                val_out, _,_,_ = self.model(data)
                val_loss  = F.nll_loss(val_out, data.y)
                val_losses.append(val_loss.item())

        val_loss = np.mean(val_losses)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        
        return val_loss
    
    def test(self):
        # test
        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data.to(self.device)
                test_out, _,_,_ = self.model(data)

                y_true += data.y.tolist()
                y_pred += test_out.max(1).indices.tolist()

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            self.logger.info("Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}".format(acc, auc, f1))
        
        return acc, auc, f1

   
    def train_process(self):

        start_time = time.time()

        for epoch in range(self.args.n_epochs):
            
            train_epoch_loss = self.train_epoch(epoch)
            
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # validation
            self.validate_epoch(epoch)
            
            self.test()
        
        if self.args.train_mode == 'pretrain':
            torch.save(self.model.state_dict(), "pretrained_models/"+self.args.pretrain_dataset+"_rprl_pretrained.pth")

        acc, auc, f1 = self.test()

        return acc, auc, f1