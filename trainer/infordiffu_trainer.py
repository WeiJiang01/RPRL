from utils.logger import (
    get_logger, 
    get_log_dir,
)
import time
import os
from utils.evaluate import Metrics
import numpy as np
from tqdm import tqdm
import torch

class InforDiffuTrainer(object):
    def __init__(self, dataset, model, optimizer, loss, args, device):
        super(InforDiffuTrainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.args = args

        self.k_list = [10, 50, 100]

        self.train_data, self.valid_data, self.test_data, self.relation_graph = dataset
        
        self.metric = Metrics()

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))


    def train_epoch(self):
        ''' Epoch operation in training phase'''
        self.model.train()

        total_loss = 0.0
        n_total_words = 0.0
        n_total_correct = 0.0
        batch_num = 0.0

        for i, batch in tqdm(enumerate(self.train_data)): # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
            # prepare data
            tgt, tgt_timestamp, tgt_idx = (item.to(self.device) for item in batch)

            np.set_printoptions(threshold=np.inf)
            gold = tgt[:, 1:]

            n_words = gold.data.ne(self.args.PAD).sum().float()
            n_total_words += n_words
            batch_num += tgt.size(0)

            self.optimizer.zero_grad()
            data = tgt, tgt_timestamp, self.relation_graph
            pred, state_probs, beta, edge_index_list = self.model(data)
            # backward
            loss, n_correct = self.get_performance(self.loss, pred, gold)
            p_loss = self.model.pde_loss(state_probs, beta, edge_index_list)
            loss = loss + self.args.alpha*torch.log(1+p_loss)
            loss.backward()

            # update parameters
            self.optimizer.step()

            # note keeping
            n_total_correct += n_correct
            total_loss += loss.item()

        return total_loss/n_total_words, n_total_correct/n_total_words


    def train_process(self):

        start_time = time.time()

        for epoch in range(self.args.epoch):
            print('\n[ Epoch', epoch, ']')

            train_loss, train_accu = self.train_epoch()

            self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_loss))

            # Valid
            if (epoch+1) % 5 == 0: 
                scores = self.evaluate_epoch(self.valid_data)
                for metric in scores.keys():
                    scores[metric] = round(scores[metric], 4)
                    self.logger.info("Test " + metric + ' ' + str(scores[metric]))
        
        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    .format(
                        (training_time / 60), 
                        ))
        
        if self.args.train_mode == 'pretrain':
            torch.save(self.model.state_dict(), "pretrained_models/"+self.args.pretrain_dataset+"_rprl_pretrained.pth")
        
        scores = self.evaluate_epoch(self.test_data)
        for metric in scores.keys():
            scores[metric] = round(scores[metric], 4)
            self.logger.info("Test " + metric + ' ' + str(scores[metric]))

        return scores


    def evaluate_epoch(self, eval_data):
        ''' Epoch operation in evaluation phase '''
        self.model.eval()
        with torch.no_grad():
            scores = {}
            for k in self.k_list:
                scores['hits@' + str(k)] = 0
                scores['map@' + str(k)] = 0

            n_total_words = 0
            for i, batch in enumerate(eval_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
                print("Evaluating batch ", i)
                # prepare data
                tgt, tgt_timestamp, tgt_idx = (item.to(self.device) for item in batch)
                y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

                # forward
                data = tgt, tgt_timestamp, self.relation_graph
                pred,_,_,_ = self.model(data)
                y_pred = pred.detach().cpu().numpy()

                scores_batch, scores_len = self.metric.compute_metric(y_pred, y_gold, self.k_list)
                n_total_words += scores_len
                for k in self.k_list:
                    scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                    scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

            for k in self.k_list:
                scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
                scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores
    
    def get_performance(self, crit, pred, gold):
        ''' Apply label smoothing if needed '''
        loss = crit(pred, gold.contiguous().view(-1))
        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        # print ("get performance, ", gold.shape, pred.shape)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(self.args.PAD).data).sum().float()
        return loss, n_correct