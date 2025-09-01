''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import pickle
from torch_geometric.data import Data
import os

class Options(object):
    def __init__(self, data_name='twitter'):
        # all dataset files for the model
        
        ######################################## Basic cascade files ########################################
        
        # all the data and corresponding id .
        self.data = 'dataset/' + data_name + '/all_cascade.txt'
        self.data_id = 'dataset/' + data_name + '/all_cascade_id.txt'

        # the cascade files with format (user1,timestamp1 user2,timestamp2 user3,timestamp3 user4,timestamp4)
        self.train_data = 'dataset/' + data_name + '/cascade.txt'                # train file path.
        self.valid_data = 'dataset/' + data_name + '/cascadevalid.txt'           # valid file path.
        self.test_data = 'dataset/' + data_name + '/cascadetest.txt'             # test file path.

        # the cascade id files (id )
        self.train_data_id = 'dataset/' + data_name + '/cascade_id.txt'          # train id file path.
        self.valid_data_id = 'dataset/' + data_name + '/cascadevalid_id.txt'     # valid id file path.
        self.test_data_id = 'dataset/' + data_name + '/cascadetest_id.txt'       # test id file path.

        # user dict and list
        self.u2idx_dict = 'dataset/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = 'dataset/' + data_name + '/idx2u.pickle'

        # user and item dict and list
        self.ui2idx_dict = 'dataset/' + data_name + '/ui2idx.pickle'
        self.idx2ui_dict ='dataset/' +  data_name + '/idx2ui.pickle'

        ######################################## Basic network files ########################################
        
        # social network file
        self.net_data = 'dataset/' +  data_name + '/edges.txt'
        
        # diffusion net file
        self.repost_net_data = 'dataset/' +  data_name + "/edges_reposts.txt"

        # Bipartite edge file
        self.item_net_data = 'dataset/' +  data_name + "/edges_item.txt"
        

        ######################################## additional files ########################################
    
        # preprocessed social network file
        self.net_data_refined = data_name + '/edges_refined.txt'
        
        ######################################## Other useless files ########################################
        
        # save path.
        self.save_path = ''

def conhyper_graphlist(cascades, timestamps, user_size, step_split):
        '''split the graph to sub graphs, return the list'''

        times, root_list = conhyper_diffsuiongraph(cascades, timestamps, user_size)
        zero_vec = torch.zeros_like(times)
        one_vec = torch.ones_like(times)
        
        time_sorted = []
        graph_list = {}
        
        for time in timestamps:
            time_sorted += time[:-1]
        time_sorted = sorted(time_sorted)
        split_length = len(time_sorted) // step_split
        
        for x in range(split_length, split_length * step_split , split_length):
            if x == split_length:
                sub_graph = torch.where(times > 0, one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
            else:
                sub_graph = torch.where(times > time_sorted[x-split_length], one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
            
            graph_list[time_sorted[x]] = sub_graph
        
        graphs = [graph_list, root_list]
        
        return graphs
        
        
def conhyper_diffsuiongraph(cascades, timestamps, user_size):
    '''return the adj. and time adj. of hypergraph'''
    e_size = len(cascades)+1
    n_size = user_size
    rows = []
    cols = []
    vals_time = []
    root_list = [0]

        
    for i in range(e_size-1):
        root_list.append(cascades[i][0])
        rows += cascades[i][:-1]
        cols +=[i+1]*(len(cascades[i])-1)
        #vals +=[1.0]*(len(cascades[i])-1)
        vals_time += timestamps[i][:-1]
        
    root_list = torch.tensor(root_list)
    Times = torch.sparse_coo_tensor(torch.Tensor([rows,cols]), torch.Tensor(vals_time), [n_size,e_size])
        
        
    return Times.to_dense(), root_list


def build_social_graph(data_name):
    options = Options(data_name)
    _u2idx = {} 

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    
    edges_list = []
    if os.path.exists(options.net_data):
        with open(options.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            if data_name == 'christianity':
                relation_list = [edge.split(' ') for edge in relation_list]
            else:
                relation_list = [edge.split(',') for edge in relation_list]

            relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if edge[0] in _u2idx and edge[1] in _u2idx]
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            edges_list += relation_list_reverse
    else:
        return [] 
    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()    
    data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)
    
    return data

def split_data(args, data_name, train_rate =0.8, valid_rate = 0.1, random_seed=0, load_dict=True, with_EOS=True, max_len = 500):
        options = Options(data_name)
        u2idx = {}
        idx2u = []
        ui2idx = {}
        idx2ui = []
        
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)
        user_size = len(u2idx)
        with open(options.ui2idx_dict, 'rb') as handle:
            ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            idx2ui = pickle.load(handle)
        # user_size = len(u2idx)
        
        user_count = [1] * user_size
        t_cascades = []
        timestamps = []
        for line in open(options.data):
            if len(line.strip()) == 0:
                continue
            timestamplist = []
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    # Twitter,Douban
                    if len(chunk.split(",")) ==2:
                        user, timestamp = chunk.split(",")
                    # Android,Christianity
                    # elif len(chunk.split())==3:
                    #     root, user, timestamp = chunk.split()                                           
                    #     if root in u2idx:          
                    #         userlist.append(u2idx[root])                        
                    #         timestamplist.append(float(timestamp))
                except:
                    print(chunk)
                if user in u2idx:
                    userlist.append(u2idx[user])
                    user_count[u2idx[user]] += 1
                    timestamplist.append(float(timestamp))
            
            # if len(userlist) > max_len:
            if len(userlist) > max_len and len(userlist) <= 500:    
                userlist = userlist[:max_len]
                timestamplist = timestamplist[:max_len]
                
            # if len(userlist) <= max_len:
            if len(userlist) >= 2 and len(userlist) <= max_len:
                if with_EOS:
                    userlist.append(args.EOS)
                    timestamplist.append(args.EOS)
                t_cascades.append(userlist)
                timestamps.append(timestamplist)
                
        # read all ids 
        t_cascades_ids = []
        for line in open(options.data_id):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip()        
            t_cascades_ids.append(ui2idx[(chunks)]) 
            
            
        
        '''ordered by timestamps'''        
        order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x:x[1])]
        timestamps = sorted(timestamps)
        t_cascades[:] = [t_cascades[i] for i in order]
        cas_idx =  [t_cascades_ids[i] for i in order]
        
        '''data split'''
        train_idx_ = int(train_rate*len(t_cascades))
        train = t_cascades[0:train_idx_]
        train_t = timestamps[0:train_idx_]
        train_idx = cas_idx[0:train_idx_]
        train = [train, train_t, train_idx]
        
        valid_idx_ = int((train_rate+valid_rate)*len(t_cascades))
        valid = t_cascades[train_idx_:valid_idx_]
        valid_t = timestamps[train_idx_:valid_idx_]
        valid_idx = cas_idx[train_idx_:valid_idx_]
        valid = [valid, valid_t, valid_idx]
        
        test = t_cascades[valid_idx_:]
        test_t = timestamps[valid_idx_:]
        test_idx = cas_idx[valid_idx_:]
        test = [test, test_t, test_idx]
            
        total_len =  sum(len(i)-1 for i in t_cascades)
        train_size = len(train_t)
        valid_size = len(valid_t)
        test_size = len(test_t)
        print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
        print("total size:%d " %(len(t_cascades)))
        print("average length:%f" % (total_len/len(t_cascades)))
        print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
        print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))    
        print("user size:%d"%(user_size-2))           
        
        return user_size, t_cascades, timestamps, train, valid, test




def build_index(data):
    user_set = set()  # 用户集合
    u2idx = {}  # u2idx[user]=pos pos是用户的编号，user是用户
    idx2u = []  # idx2u[pos]=user

    lineid = 0
    for line in open(data):
        lineid += 1
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    user_set.add(root)
            except:
                print(line)
                print(chunk)
                print(lineid)
            user_set.add(user)
    pos = 0
    u2idx['<blank>'] = pos
    idx2u.append('<blank>')
    pos += 1
    u2idx['</s>'] = pos
    idx2u.append('</s>')
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        idx2u.append(user)
        pos += 1
    user_size = len(user_set) + 2
    print("user_size : %d" % (user_size))
    return user_size, u2idx, idx2u


class InforDiffuDataLoader(object):
    ''' For data iteration ''' 

    def __init__(self, args, cas, batch_size=64, load_dict=True, cuda=True,  test=False, with_EOS=True,_need_shsuffle=False, max_len =500): 
        self._batch_size = batch_size
        self.cas = cas[0]
        self.time = cas[1]
        self.idx = cas[2]
        self.test = test
        self.with_EOS = with_EOS          
        self.cuda = cuda
        self.args = args
        self._need_shuffle = _need_shsuffle
        self.max_len = max_len 
        
        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

        #  regroup the training data 
        if self._need_shuffle:
            num = [x for x in range(0, len(self.cas))]
            random_seed_int = random.randint(0, 1000)
            print(f"Init Dataset and shuffle data with {random_seed_int}")
            random.seed(random_seed_int)
            random.shuffle(num)
            self.cas = [self.cas[num[i]] for i in range(0, len(num))]
            self.time = [self.time[num[i]] for i in range(0, len(num))]
            self.idx = [self.idx[num[i]] for i in range(0, len(num))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            max_len = 205
            inst_data = np.array([
                inst + [self.args.PAD] * (max_len - len(inst))
                for inst in insts])
                
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            if self.cuda:
                seq_idx = Variable(torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test).cuda()
            else :
                seq_idx = Variable(torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)
            return seq_data, seq_data_timestamp, seq_idx
        else:
            if self._need_shuffle:
                num = [x for x in range(0, len(self.cas))]
                random_seed_int = random.randint(0, 1000)
                print(f"shuffle data with {random_seed_int}")
                random.seed(random_seed_int)
                random.shuffle(num)
                self.cas = [self.cas[num[i]] for i in range(0, len(num))]
                self.time = [self.time[num[i]] for i in range(0, len(num))]
                self.idx = [self.idx[num[i]] for i in range(0, len(num))]


            self._iter_count = 0
            raise StopIteration()
