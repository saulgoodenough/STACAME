import numpy as np
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv
from typing import List, Optional, Union

import torch
from torch import nn
import random


class DomainSpecificBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_domains=2):
        super().__init__()
        self.num_domains = num_domains
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(num_features) for _ in range(num_domains)
        ])

    def forward(self, x, domain_id):
        # x: [B, C], domain_id: scalar or tensor of domain indices
        if isinstance(domain_id, int):
            return self.bn_layers[domain_id](x)
        elif isinstance(domain_id, torch.Tensor):
            # e.g., domain_id = tensor of shape [B]
            out = torch.zeros_like(x)
            for d in range(self.num_domains):
                idx = (domain_id == d).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    out[idx] = self.bn_layers[d](x[idx])
            return out
        else:
            raise ValueError("Unsupported domain_id format.")




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

# class RBF(nn.Module):
#     """
#     Radial Basis Function (RBF) kernel for MMD loss.
#
#     Computes a mixture of Gaussian kernels with different bandwidths.
#     The multipliers are created lazily to avoid device initialisation
#     issues at module import time.
#     """
#
#     def __init__(
#         self,
#         n_kernels: int = 5,
#         mul_factor: float = 2.0,
#         bandwidth: Optional[float] = None,
#         device: torch.device = torch.device(
#             "cuda:0" if torch.cuda.is_available() else "cpu"
#         ),
#     ):
#         super().__init__()
#         self.n_kernels = n_kernels
#         self.mul_factor = mul_factor
#         self.bandwidth = bandwidth
#         self.device = device
#         # Multipliers will be built on the first forward pass
#         self.bandwidth_multipliers = None
#
#     def _init_multipliers(self):
#         """Create the bandwidth multipliers on the correct device."""
#         if self.bandwidth_multipliers is None:
#             self.bandwidth_multipliers = (
#                 self.mul_factor
#                 ** (torch.arange(self.n_kernels) - self.n_kernels // 2)
#             ).to(self.device)
#
#     def get_bandwidth(self, L2_distances: torch.Tensor) -> torch.Tensor:
#         """Heuristic bandwidth estimation from median pairwise distance."""
#         n = L2_distances.shape[0]
#         if self.bandwidth is None:
#             # Use median of the upper‑triangular distances
#             return (
#                 L2_distances[torch.triu_indices(n, n, 1)].median()
#                 / np.log(n)
#             )
#         return self.bandwidth
#
#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         Compute multi‑bandwidth RBF kernel values.
#
#         Parameters
#         ----------
#         X : torch.Tensor
#             Input tensor of shape (batch_size, feature_dim).
#
#         Returns
#         -------
#         torch.Tensor
#             Kernel values of shape (n_kernels, batch_size, batch_size).
#         """
#         self._init_multipliers()
#         L2_distances = (torch.cdist(X, X) ** 2).to(self.device)
#         bw = self.get_bandwidth(L2_distances)
#         scales = bw * self.bandwidth_multipliers[:, None, None]
#         return torch.exp(-L2_distances[None, ...] / scales).sum(dim=0)
#
# class MMDLoss(nn.Module):
#     def __init__(self, kernel=RBF(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')),
#                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
#         super().__init__()
#         self.kernel = kernel
#         self.device = device
#
#     def forward(self, X, Y):
#         device = self.device
#         K = self.kernel(torch.vstack([X, Y]).to(device)).to(device)
#
#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean().to(device)
#         XY = K[:X_size, X_size:].mean().to(device)
#         YY = K[X_size:, X_size:].mean().to(device)
#         return (XX - 2 * XY + YY).to(device)



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
        

class STACAME_light(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME_light, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(out_dim, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        #h2 = self.conv2(h1, edge_index, attention=False)
        # self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        # self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        
        h4 = self.conv4(h1, edge_index, attention=False)

        return h1, h4  # F.log_softmax(x, dim=-1)

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
        
        # 可选的MLP后处理
        if self.use_mlp:
            h4 = h4 + self.mlp(h4)  # 残差连接
            
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




class STACAME_minibatch_large(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME_minibatch_large, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
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
                    h1 = F.elu(self.conv1(features, edge_index))
                #elif i == 1:
                    h2 = self.conv2(h1, edge_index, attention=False)
                    self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
                    self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
                    self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
                    self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
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
            self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
            self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
            h3 = F.elu(self.conv3(h2, adjs, attention=True,
                                  tied_attention=self.conv1.attentions))
            h4 = self.conv4(h3, adjs, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)
        
    def inference(self, x_all, all_loader):
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

        return h2, h4  # F.log_softmax(x, dim=-1)
        
    def inference(self, x_all, all_loader):
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



class WDiscriminator(torch.nn.Module):
    r"""
    WGAN Discriminator
    
    Parameters
    ----------
    hidden_size
        input dim
    hidden_size2
        hidden dim
    """
    def __init__(self, hidden_size:int, hidden_size2:Optional[int]=512):
        super(WDiscriminator, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)
    def forward(self, input_embd):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))



class STACAME_Multi(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME_Multi, self).__init__()
        #self.FCN_list = []
        #for spe_id, spe_input_dim in self.species_dim_dict.items():
        
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.in_dim = in_dim
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features_dict, edge_index, device):
        #features_0 = features[0:]    
        s = 0
        for species_id in features_dict.keys():
            if s == 0:
                features = F.leaky_relu(nn.Linear(features_dict[species_id].shape[1], self.in_dim).to(device)(features_dict[species_id].to(device)))
            else:
                features_temp = F.leaky_relu(nn.Linear(features_dict[species_id].shape[1], self.in_dim).to(device)(features_dict[species_id].to(device)))
                features = torch.concat((features, features_temp), axis=0).to(device)
            s += 1
        #features = features.to(device)

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4, features  # F.log_softmax(x, dim=-1)


class STAligner(torch.nn.Module):
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


class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)

class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512
        self.attention = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Sigmoid()  # 输出 [0, 1] 权重
        )

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

        self.feature_mask = nn.Parameter(torch.ones(n_latent))  # 可训练参数

    def forward(self, z):
        # attn_weights = self.attention(z)
        # z = z * attn_weights
        z = z * self.feature_mask
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x

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
