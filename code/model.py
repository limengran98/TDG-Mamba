import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv,  GATConv, SAGEConv, DirGNNConv


class TDG_Mamba(torch.nn.Module):
    def __init__(self, args, num_features):
        super(TDG_Mamba, self).__init__()

        num_features = num_features
        hidden_channels = args.hidden_channels 
        d_model = args.d_model
        d_state = args.d_state
        d_conv = args.d_conv
        expand = args.expand
        device = args.device
        self.device = device
        self.gcn = DirGNN(num_features, hidden_channels, args).to(device)
        self.self_attn = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).to(device)
        self.hybrid_layer_norm = torch.nn.LayerNorm(num_features)

    def forward(self, x, input_ids, attention_mask,edge_index):
        inv_x = torch.flip(x, [1])
        x_attention = self.self_attn(x)
        inv_x_attention = self.self_attn(inv_x)
        x = F.relu(torch.sum(x_attention + torch.flip(inv_x_attention, [1]), dim=-1))
        x = self.hybrid_layer_norm(x)
        x = self.gcn(x, edge_index)     
        return x


class DirGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, args):
        super(DirGNN, self).__init__()
        if args.MPNN =='GCN':
            self.conv1 = DirGNNConv(GCNConv(num_features, hidden_channels))
            self.conv2 = DirGNNConv(GCNConv(hidden_channels, hidden_channels))
        if args.MPNN =='GAT':
            self.conv1 = DirGNNConv(GATConv(num_features, hidden_channels))
            self.conv2 = DirGNNConv(GATConv(hidden_channels, hidden_channels))
        if args.MPNN =='GraphSAGE':
            self.conv1 = DirGNNConv(SAGEConv(num_features, hidden_channels))
            self.conv2 = DirGNNConv(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = args.dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
