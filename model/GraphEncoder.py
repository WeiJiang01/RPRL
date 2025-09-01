import sys,os
sys.path.append(os.getcwd())
from dataloaders.rumor_dataloader import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.nn import BatchNorm1d

    

class GraphEncoder(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, args, device):
        super(GraphEncoder, self).__init__()
        hidden = args.graph_hidden_dim

        self.device = device
        self.fc_residual = True  # no skip-connections for fc layers.
        self.conv_residual = True
        self.gconv_dropout = args.gconv_dropout
        hidden_in = args.num_features

        if args.task == 'socialbot_detection':
            num_relations = args.num_relations
        else:
            num_relations = 1

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = RGCNConv(hidden_in, hidden, num_relations=num_relations)
        
        self.gating = None
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        
        for i in range(args.n_layers_conv):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(RGCNConv(hidden, hidden, num_relations=num_relations))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, x, edge_index, edge_type):

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, edge_type))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, edge_type))
            x = x + x_ if self.conv_residual else x_
        if self.gconv_dropout > 0:
            x = F.dropout(x, p=self.gconv_dropout, training=self.training)

        return x

    def __repr__(self):
        return self.__class__.__name__

