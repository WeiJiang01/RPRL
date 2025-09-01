import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model.GraphEncoder import GraphEncoder
import torch.nn.init as init
from torch_geometric.nn import global_add_pool
import math
from torch.autograd import Variable
import numpy as np
from model.Transformer import TransformerBlock
from model.Layers import *



class RPRL(nn.Module):
    def __init__(self, args=None, device=None):
        super(RPRL, self).__init__()
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.p_num_heads = args.p_num_heads
        self.p_num_layers = args.p_num_layers
        hidden_dim = args.hidden_dim
        num_features = args.num_features
        self.task = args.task
        self.device = device
        self.args = args
        self.hidden_dim = hidden_dim
        self.pos_dim = 8
        self.g_c = args.g_c

        self.projection_layer = torch.nn.Linear(num_features, hidden_dim)

        self.graph_encoder = GraphEncoder(args=args, device=device).to(device)

        if args.task == 'information_diffusion':
            hidden_dim = hidden_dim + self.pos_dim
            self.p_hidden_dim = args.p_hidden_dim + self.pos_dim
        else:
            hidden_dim = hidden_dim
            self.p_hidden_dim = args.p_hidden_dim

        self.tf_blocks = nn.ModuleList(
            [TransformerBlock(d_model=hidden_dim, n_heads=self.num_heads, attn_dropout=args.trans_dropout) for _ in range(self.num_layers)]
        )


        self.pg_att_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.p_hidden_dim, n_heads=self.p_num_heads, attn_dropout=args.trans_dropout) for _ in range(self.p_num_layers)]
        )

        self.pg_out = nn.Linear(self.p_hidden_dim, 3)
        self.coef_layer = nn.Linear(3, 2)

        self.fusion = Fusion(hidden_dim)
        self.decode_layer = torch.nn.Linear(hidden_dim, hidden_dim)

        self.global_pool = global_add_pool

        if args.task == 'rumor_detection':
            num_classes = args.num_classes

            self.fc = torch.nn.Linear(hidden_dim, num_classes)
        elif args.task == 'socialbot_detection':
            num_classes = args.num_classes

            self.fc = torch.nn.Linear(hidden_dim, num_classes)

        elif args.task == 'information_diffusion':
            self.n_node = args.user_size
            self.pos_embedding = nn.Embedding(1000, self.pos_dim)
            self.embedding = nn.Embedding(self.n_node, num_features, padding_idx=0)
            
            if args.train_mode == 'finetune':
                state = torch.load("pretrained_models/"+args.pretrain_dataset+"_rprl_pretrained.pth", map_location=device)
                state.pop("embedding.weight", None)
                state.pop("pos_embedding.weight", None)
                self.load_state_dict(state, strict=False)

            self.projection_layer_IDP = EmbedInit(num_features, self.hidden_dim, dropout=0.3)

            self.fc = nn.Linear(self.hidden_dim+self.pos_dim, self.n_node)
            self.dropout = nn.Dropout(0.3)
            self.reset_parameters()
            self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.embedding.weight)
        

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def combine_seq(self, x, batch):
        batch_num = batch.max().item() + 1  
        grouped_x = [x[batch == i] for i in range(batch_num)]  #

        x_padded = pad_sequence(grouped_x, batch_first=True, padding_value=0)  # (batch_num, max_length, H)

        return x_padded
    
    def pad_hop_labels(self, hop_label, batch, pad_value=-1):

        batch_size = int(batch.max().item()) + 1
        
        hop_labels_list = []
        for i in range(batch_size):
            indices = (batch == i).nonzero(as_tuple=True)[0]
            hop_labels_list.append(hop_label[indices])
            
        max_len = max(hl.size(0) for hl in hop_labels_list)
        
        padded_list = []
        for hl in hop_labels_list:
            padded = torch.full((max_len,), pad_value, dtype=hl.dtype, device=hl.device)
            padded[:hl.size(0)] = hl  
            padded_list.append(padded)
        
        padded_hop = torch.stack(padded_list)

        return padded_hop


    def hop_masking_rumor(self, x, padded_hop_label, max_hop, pad_value=-1):
   
        t = max_hop

        time_range = torch.arange(t, device=x.device).view(1, t, 1)
        
        hop_label_expanded = padded_hop_label.unsqueeze(1)
        
        mask = ((hop_label_expanded != pad_value) & (time_range >= hop_label_expanded)).float()

        x_time = x.unsqueeze(1) * mask.unsqueeze(-1)
        
        return x_time
    

    def split_edge_index_rumor(self, edge_index, batch):
        batch_num = batch.max().item() + 1
        edge_index_list = []

        for i in range(batch_num):
            node_mask = (batch == i)
            node_indices = torch.nonzero(node_mask, as_tuple=False).view(-1)

            old_to_new_map = -torch.ones(batch.size(0), dtype=torch.long, device=batch.device)
            old_to_new_map[node_indices] = torch.arange(node_indices.size(0), device=batch.device)

            src, dst = edge_index
            src_mask = node_mask[src]
            dst_mask = node_mask[dst]
            edge_mask = src_mask & dst_mask
            sub_edge_index = edge_index[:, edge_mask]

            remapped_edge_index = old_to_new_map[sub_edge_index]

            edge_index_list.append(remapped_edge_index)

        return edge_index_list
    
    def split_edge_index_infordiff(self, input_tensor):
       
        edge_index_list = []
        for row in input_tensor:
            nz = torch.nonzero(row != 0, as_tuple=False)
            if nz.numel() == 0:
                edge_index_list.append(torch.empty((2, 0), dtype=torch.long, device=row.device))
                continue

            last_idx = nz.max().item()
            valid_len = last_idx + 1

            if valid_len < 2:
                edge_index_list.append(torch.empty((2, 0), dtype=torch.long, device=row.device))
                continue

            src = torch.arange(valid_len - 1, device=row.device)
            tgt = torch.arange(1, valid_len, device=row.device)
            edge_index = torch.stack([src, tgt], dim=0)  # shape = [2, valid_len-1]
            edge_index_list.append(edge_index)

        return edge_index_list
    
    
    def hop_masking_socialbot(self, x_data_n, hop_labels, n_hops):
       
        num_nodes, num_all, hidden_dim = x_data_n.shape

        x_masked = torch.zeros(
            (num_nodes, n_hops, num_all, hidden_dim),
            dtype=x_data_n.dtype,
            device=x_data_n.device
        )

        for h in range(n_hops):
            keep = (hop_labels <= (h + 1)).unsqueeze(-1)
            x_masked[:, h, :, :] = x_data_n * keep

        return x_masked

    def hop_masking_infordiff(self, x, T):
        
        B, N, H = x.shape
        device = x.device

        x_t = x.unsqueeze(1).repeat(1, T, 1, 1)  # (B, T, N, H)

        time_idxs = torch.arange(T, device=device).unsqueeze(1)   # (T, 1)
        node_idxs = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
        mask_base = (node_idxs <= time_idxs).to(x.dtype)         

        mask = mask_base.unsqueeze(0).unsqueeze(-1)               # (1, T, N, 1)

        return x_t * mask                                        # (B, T, N, H)

    def forward(self, data):
        # ----------- init data -------------------
        if self.task == 'rumor_detection':
            x, edge_index, batch, hop_label = data.x, data.edge_index, data.batch, data.hop_label

            select_hop = self.args.select_hop

            init_x = x

            x = self.combine_seq(x, batch)

            padded_hop_labels = self.pad_hop_labels(hop_label, batch, pad_value=-1)

            num_edges = edge_index.shape[1]
            edge_type = torch.zeros(num_edges, dtype=torch.long).to(self.device)
            x = F.relu(self.projection_layer(x))


        elif self.task == 'socialbot_detection':
            x, edge_index, neighbor_matrix, edge_type, hop_labels, edge_index_list = data
            if x.shape[-1] < self.args.num_features:
                x = F.pad(x, (0, self.args.num_features-x.shape[-1]))
                neighbor_matrix = F.pad(neighbor_matrix, (0, self.args.num_features-neighbor_matrix.shape[-1]))
            init_x = x

            select_hop = self.args.select_hop

            x = neighbor_matrix
            x = F.relu(self.projection_layer(x))
        
        elif self.task == 'information_diffusion':
            input, input_timestamp, graph = data
            input = input[:, :-1]

            input_timestamp = input_timestamp[:, :-1]
            hidden = (self.projection_layer_IDP(self.embedding.weight, graph.to(self.device)))
            hidden = self.dropout(hidden)
            num_edges = graph.edge_index.shape[1]
            edge_type = torch.zeros(num_edges, dtype=torch.long).to(self.device)

            edge_index = graph.edge_index.to(self.device)

            init_x = (self.embedding.weight)

            mask = (input == self.args.PAD)
            batch_t = torch.arange(input.size(1)).expand(input.size()).to(self.device)
            order_embed = self.dropout(self.pos_embedding(batch_t))
            batch_size, max_len = input.size()

            dyemb = torch.zeros(batch_size, max_len, self.hidden_dim).to(self.device)
            dyemb = F.embedding(input.to(self.device), hidden.to(self.device))

            x = torch.cat([dyemb, order_embed], dim=-1).to(self.device)

            select_hop = self.args.select_hop

            edge_index_list = self.split_edge_index_infordiff(input)
            
        # ----------- encoding -------------------

        for block in self.tf_blocks:
            x = block(x,x,x)

        g_x = self.graph_encoder(init_x, edge_index, edge_type)
        
        # ----------- Pg Attention -------------------
        if self.task == 'rumor_detection':
            x_t = self.hop_masking_rumor(x, padded_hop_labels, select_hop)
        if self.task == 'socialbot_detection':
            x_t = self.hop_masking_socialbot(x, hop_labels, select_hop)
        if self.task == 'information_diffusion':
            x_t = self.hop_masking_infordiff(x, select_hop)

        x_t = x_t.permute(0, 2, 1, 3)

        batch_size = x_t.shape[0]
        num_node = x_t.shape[1]
        x_t = x_t.reshape(batch_size * num_node, select_hop, self.p_hidden_dim)


        for block in self.pg_att_blocks:
            x_t = block(x_t, x_t, x_t)


        x_t = x_t.reshape(batch_size, num_node, select_hop, self.p_hidden_dim)
        
        x_p = x_t[:,:,-1,:]

        x = x + F.relu(x_p)

        state_logits = self.pg_out(x_t)

        state_probs = F.softmax(state_logits, dim=-1)

        if self.args.task == 'rumor_detection':
            edge_index_list = self.split_edge_index_rumor(edge_index, batch)

        beta = F.sigmoid(self.coef_layer(state_probs).sum(dim=[0,1,2]))

        # ----------- decoding -------------------

        if self.task == 'rumor_detection':
            g_x = self.combine_seq(g_x, batch)
            g_x = g_x.sum(dim=1)
            x = self.fusion((1-self.g_c)*x[:,0,:], self.g_c*g_x)
            x = F.relu(self.decode_layer(x))
            x = self.fc(x)

            out = F.log_softmax(x, dim=1)

        elif self.task == 'socialbot_detection':
            x = (x[:,0,:].squeeze(1))

            x = self.fusion((1-self.g_c)*x, self.g_c*g_x)
            x = F.leaky_relu(self.decode_layer(x))
            out = self.fc(x)
        
        elif self.task == 'information_diffusion':
            x = self.dropout(x)
            g_x = F.embedding(input.to(self.device), g_x.to(self.device))
            g_x = torch.cat([g_x, order_embed], dim=-1).cuda()
            x = self.fusion((1-self.g_c)*x, self.g_c*g_x)

            output_u = self.fc(x.to(self.device)) 
            mask = self.get_previous_user_mask(input.cpu(), self.n_node)
            out = (output_u + mask).view(-1, output_u.size(-1)).to(self.device)

        return out, state_probs, beta, edge_index_list

    def init_optimizer(self, args):
  
        for p in self.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        return optimizer
    
    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.to(self.device)
        masked_seq = previous_mask * seqs.data.float()

        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.to(self.device)
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.to(self.device)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
        masked_seq = Variable(masked_seq, requires_grad=False)
        
        return masked_seq.to(self.device)
    
    def state_matching_loss(self, state_prob, state_label):
        pass

    def pde_loss(self, state_probs, beta, edge_index_list, discrete_time_deriv=True):
      
        device = state_probs.device
        B, N, T, D = state_probs.shape  # D = 1 + J
        U = state_probs[..., 0]          # (B, N, T)
        I = state_probs[..., 1:]         # (B, N, T, J)
        J = D - 1

        if not discrete_time_deriv or T <= 1:
            return torch.tensor(0.0, device=device)

        offsets = torch.arange(B, device=device) * N
        src_list = [edge_index_list[b][0].to(device) + offsets[b] for b in range(B)]
        tgt_list = [edge_index_list[b][1].to(device) + offsets[b] for b in range(B)]
        src_global = torch.cat(src_list, dim=0)  
        tgt_global = torch.cat(tgt_list, dim=0)

        I_flat = I.reshape(B * N, T, J)
        agg_flat = torch.zeros_like(I_flat, device=device)  # (B*N, T, J)
        agg_flat.index_add_(0, tgt_global, I_flat[src_global])
        agg_flat.index_add_(0, src_global, I_flat[tgt_global])
        aggregated = agg_flat.view(B, N, T, J)  # (B, N, T, J)

        dU_dt = U[:, :, 1:] - U[:, :, :-1]           # (B, N, T-1)
        dI_dt = I[:, :, 1:, :] - I[:, :, :-1, :]     # (B, N, T-1, J)
        U_t    = U[:, :, :-1]                        # (B, N, T-1)
        agg_t  = aggregated[:, :, :-1, :]            # (B, N, T-1, J)

        term      = U_t.unsqueeze(-1) * agg_t * beta # (B, N, T-1, J)
        sum_over_j= term.sum(dim=-1)                 # (B, N, T-1)
        PDE_U     = dU_dt - sum_over_j               # (B, N, T-1)
        PDE_I     = dI_dt - term                     # (B, N, T-1, J)

        loss = PDE_U.pow(2).sum() + PDE_I.pow(2).sum()

        return loss
    
   