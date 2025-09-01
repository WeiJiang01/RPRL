import torch
from torch.utils.data import Dataset


class SocialBotDataLoader(Dataset):
    def __init__(self, dataset, args, device='cpu'):
        self.device = device
        self.dataset = dataset
        self.args = args

    def load_data(self):
        if self.dataset == 'twibot-22':

            self.root = 'dataset/twibot-22/'

            labels=torch.load(self.root+"label.pt").to(self.device)
            des_tensor=torch.load(self.root+"des_tensor.pt").to(self.device)
            
            tweets_tensor=torch.load(self.root+"tweets_tensor.pt").to(self.device)
            
            num_prop=torch.load(self.root+"num_properties_tensor.pt").to(self.device)
            
            category_prop=torch.load(self.root+"cat_properties_tensor.pt").to(self.device)
            
            edge_index=torch.load(self.root+"edge_index.pt").to(self.device)
            edge_type=torch.load(self.root+"edge_type.pt").to(self.device)
        
            train_idx=torch.load(self.root+'train_idx.pt')
            val_idx=torch.load(self.root+'val_idx.pt')
            test_idx=torch.load(self.root+'test_idx.pt')
                
            x_data = torch.cat((des_tensor,tweets_tensor,num_prop,category_prop),dim=1)

        elif self.dataset == 'MGTAB':
            self.root = 'dataset/MGTAB/'

            labels = torch.load(self.root+"label.pt").to(self.device)
            edge_index = torch.load(self.root+"edge_index.pt").to(self.device)
            edge_index = torch.tensor(edge_index, dtype = torch.int64).to(self.device)
            edge_type = torch.load(self.root+"edge_type.pt").to(self.device)
            x_data = torch.load(self.root+"features.pt").to(self.device)

            relation_dict = {
            0:'followers',
            1:'friends',
            2:'mention',
            3:'reply',
            4:'quoted',
            5:'url',
            6:'hashtag'
            }

            index_select_list = (edge_type == 100)

            if self.args.dataset == 'twibot-22':
                relation_list = [0,1,2,3]
            else:
                relation_list = [0,1,2,3,4,5,6]


            print('relation used:', end=' ')
            for features_index in relation_list:
                    index_select_list = index_select_list + (features_index == edge_type)
                    print('{}'.format(relation_dict[features_index]), end='  ')
            print('\n')
            edge_index = edge_index[:, index_select_list]
            edge_type = edge_type[index_select_list]
            # edge_type = torch.where(edge_type == 2, 0, 1)  # 2 变 0，3 变 1

            # edge_weight =  edge_weight[index_select_list]

            print(torch.max(edge_index))

            length_data = x_data.size(0)

            indices = torch.randperm(length_data)
            # indices = torch.arange(length_data)
            
            train_size = int(0.7 * length_data)
            val_size = int(0.2 * length_data)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        return [x_data, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
    
