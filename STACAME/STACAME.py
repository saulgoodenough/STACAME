import numpy as np
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
import torch.nn.functional as F
from .gat_conv import GATConv
from typing import List, Optional, Union
import torch
from torch import nn
import random


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).to(device)
        self.bandwidth = bandwidth#.to(device)
        self.device = device

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return (L2_distances.data.sum() / (n_samples ** 2 - n_samples)).to(self.device)
        return self.bandwidth

    def forward(self, X):
        L2_distances = (torch.cdist(X, X) ** 2).to(self.device)
        return (torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)).to(self.device)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')),
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.kernel = kernel
        self.device = device

    def forward(self, X, Y):
        device = self.device
        K = self.kernel(torch.vstack([X, Y]).to(device)).to(device)

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean().to(device)
        XY = K[:X_size, X_size:].mean().to(device)
        YY = K[X_size:, X_size:].mean().to(device)
        return (XX - 2 * XY + YY).to(device)



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

seed_everything(42)

class STACAME(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)
        

class STACAMEDecoder_light(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAMEDecoder_light, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = F.elu(self.conv2(h1, edge_index, attention=False))
        h3 = self.conv3(h2, edge_index, attention=False)
        return h2, h3


class STACAME_Decoder(torch.nn.Module):
    def __init__(self, hidden_dims, use_mlp=False):
        super(STACAME_Decoder, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        
        # 原有的GAT层
        self.conv1 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        
        # 可选的全连接层
        if use_mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, num_hidden),
                torch.nn.ELU(),
                torch.nn.Linear(num_hidden, in_dim)
            )
        self.use_mlp = use_mlp
        
    def forward(self, h2, edge_index):
        h1 = F.elu(self.conv1(h2, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)
        
        if self.use_mlp:
            h4 = h4 + self.mlp(h4)
            
        return h2, h4


class STACAME_minibatch(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME_minibatch, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(out_dim, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, adjs, mode="batch"):
        if mode == "batch":
            for i, (edge_index, e_id, size) in enumerate(adjs):
                # Extract target node features
                if i==0:
                    h1 = F.elu(self.conv1(features, edge_index))
                elif i == 1:
                    h4 = self.conv4(h1, edge_index, attention=False)  
        else:
            h1 = F.elu(self.conv1(features, adjs))
            #h2 = F.elu(self.conv2(features, adjs, attention=False))
            h4 = self.conv4(h1, adjs, attention=False)

        return h1, h4  # F.log_softmax(x, dim=-1)
        
    def inference(self, x_all, all_loader, device):
        # This function will be called in test
        for i in range(2):
            xs = []
            zs = []
            for batch_size, n_id, adj in all_loader:
                edge_index, _, size = adj
                features = x_all[n_id, :].to(device)
                if i==0:
                    h1 = F.elu(self.conv1(features, edge_index.to(device)))
                    zs.append(h1.cpu())
                elif i == 1:
                    h1 = F.elu(self.conv1(features, edge_index.to(device)))
                    x = self.conv4(h1, edge_index.to(device), attention=False)  
                    # Append the node embeddings to xs
                    xs.append(x.cpu())
            # Concat all embeddings into one tensor
            z_all = torch.cat(zs, dim=0)
            x_all = torch.cat(xs, dim=0)
        return z_all, x_all



class STACAMEDecoder_minibatch(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAMEDecoder_minibatch, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        # self.conv1_1 = GATConv(num_hidden, num_hidden, heads=1, concat=False,
        #                        dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, adjs, mode="batch"):
        if mode == "batch":
            for i, (edge_index, e_id, size) in enumerate(adjs):
                # Extract target node features
                if i==0:
                    #x_target = features[:size[1]]
                    h1 = F.elu(self.conv1(features, edge_index))
                #elif i == 1:
                    h2 = self.conv2(h1, edge_index, attention=False)
                    self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
                    self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
                    # self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
                    # self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
                #elif i == 2:
                    h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                                  tied_attention=self.conv1.attentions))
                #elif i == 3:
                    h4 = self.conv4(h3, edge_index, attention=False)
               
        else:
            h1 = F.elu(self.conv1(features, adjs))
            h2 = self.conv2(h1, adjs, attention=False)
            self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
            self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
            h3 = F.elu(self.conv3(h2, adjs, attention=True,
                                  tied_attention=self.conv1.attentions))
            h4 = self.conv4(h3, adjs, attention=False)

        return h2, h4
        
    def inference(self, x_all, all_loader, device):
        # This function will be called in test
        for i in range(4):
            xs = []
            for batch_size, n_id, adj in all_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                # Append the node embeddings to xs
                xs.append(x.cpu())

            # Concat all embeddings into one tensor
            x_all = torch.cat(xs, dim=0)



class STACAME_lightdecoder_minibatch(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME_lightdecoder_minibatch, self).__init__()
        in_dim, num_hidden, out_dim = hidden_dims
        self.conv1 = GATConv(out_dim, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, adjs, mode="batch"):
        if mode == "batch":
            for i, (edge_index, e_id, size) in enumerate(adjs):
                if i == 0:
                    h1 = F.elu(self.conv1(features, edge_index))
                    h2 = self.conv3(h1, edge_index, attention=True)
                return h1[:size[1]], h2[:size[1]]
        else:
            h1 = F.elu(self.conv1(features, adjs))
            h2 = self.conv3(h1, adjs, attention=True)
            return h1, h2

    def inference(self, x_all, subgraph_loader):
        h1_list = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj
            x = x_all[n_id]
            h = F.elu(self.conv1(x, edge_index))
            h1_list.append(h[:size[1]].cpu())
        h1_all = torch.cat(h1_list, dim=0)

        h2_list = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj
            h = self.conv3(h1_all[n_id], edge_index, attention=True)
            h2_list.append(h[:size[1]].cpu())
        h2_all = torch.cat(h2_list, dim=0)

        return h1_all, h2_all



class MultiClassDiscriminator(nn.Module):
    def __init__(self, n_input, num_classes):
        super(MultiClassDiscriminator, self).__init__()
        self.n_input = n_input
        self.num_classes = num_classes
        n_hidden = 256

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        # self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        # self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(num_classes, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(num_classes).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        #h = F.relu(F.linear(h, self.W_2, self.b_2))
        logits = F.linear(h, self.W_3, self.b_3)
        return logits
