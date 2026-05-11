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

# class STACAMEDecoder_light(torch.nn.Module):
#     def __init__(self, hidden_dims):
#         super(STACAMEDecoder_light, self).__init__()
#         [in_dim, num_hidden, out_dim] = hidden_dims
#         self.conv1 = GATConv(out_dim, out_dim, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv3 = GATConv(out_dim, in_dim, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#     def forward(self, features, edge_index):
#         h1 = F.elu(self.conv1(features, edge_index))
#         h2 = self.conv3(h1, edge_index, attention=True)
#         return h1, h2

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class STACAME_lightDecoder(torch.nn.Module):
    def __init__(self, hidden_dims, use_mlp=False, checkpointing=True):
        super().__init__()
        in_dim, num_hidden, out_dim = hidden_dims

        # ---- 原有 GAT 层，但不为 conv3 创建独立参数 ----
        self.conv1 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        # conv3 只保留一个空壳，实际运算会复用 conv2 的权重转置
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        # 删除 conv3 自带的参数，真正共享 conv2 的权重
        del self.conv3.lin_src
        del self.conv3.lin_dst
        # 注册一个缓冲表示 conv3 无独立参数（便于 state_dict 干净）
        self.conv3.register_buffer("_dummy", torch.tensor(0))

        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        # 可选的轻量 MLP
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, num_hidden, bias=False),
                torch.nn.ELU(),
                torch.nn.Linear(num_hidden, in_dim, bias=False)
            )
        self.checkpointing = checkpointing

    @staticmethod
    def _conv3_forward(conv2_lin_src, conv2_lin_dst, x, edge_index,
                       tied_attention, conv1_alpha):
        """
        使用 conv2 的权重转置完成 conv3 的计算，无额外参数。
        这里直接调用 GATConv 的底层消息传递，或者用 functional 实现。
        为了简洁，采用手动消息传递实现（仅支持 add_self_loops=False）。
        tied_attention: [E, 1] 或 [E, heads]
        conv1_alpha: 可选，对应 conv1 的负斜率（此处共享）
        """
        # 权重转置
        weight_src = conv2_lin_src.t()   # 形状适配 conv3 的输入->输出
        weight_dst = conv2_lin_dst.t()

        # 源/目标特征线性变换
        src = x @ weight_src          # [N, num_hidden]
        dst = x @ weight_dst          # [N, num_hidden]

        # 计算 attention logits（复用注意力系数）
        alpha = tied_attention        # [E, 1]
        # 注意：tied_attention 已经是经过 LeakyReLU 的 logits，此处省略二次计算
        # 直接使用 softmax（实际在 GATConv 内部对邻居做 softmax，这里简化）
        # 但为了与原 GATConv 对齐，利用 alpha 作为注意力权重，无需再算 softmax
        # 若 tied_attention 尚未归一化，可在此处对每条边做 softmax（取决于原实现）
        from torch_geometric.utils import softmax as pyg_softmax
        alpha = pyg_softmax(alpha, edge_index[1])   # 对目标节点做 softmax

        # 消息传递：src[edge_index[0]] * alpha + dst[edge_index[1]]
        msg = src[edge_index[0]] + dst[edge_index[1]]
        out = msg * alpha
        # 聚合（scatter）
        out = torch.zeros_like(x).scatter_add(0,
                                              edge_index[1].unsqueeze(-1).expand_as(out),
                                              out)
        # 注意：原 GATConv 可能没有最后的线性变换（concat=False, heads=1）
        # 输出就是聚合结果
        return out

    def forward(self, h2, edge_index):
        # ---- conv1 ----
        if self.checkpointing:
            # 使用 checkpoint 不保存中间激活
            h1 = checkpoint(self.conv1, h2, edge_index, use_reentrant=False)
        else:
            h1 = self.conv1(h2, edge_index)
        h1 = F.elu(h1)

        # ---- conv2 (无注意力) ----
        h2_new = self.conv2(h1, edge_index, attention=False)
        # 此时 h2 可释放（若不需保留原始输入）
        del h1

        # ---- conv3（共享 conv2 权重，复用 conv1 的注意力） ----
        # 取出 conv1 保存的注意力系数（在 checkpoint 内已被丢弃，需通过外部获取）
        # 若开启 checkpoint，conv1.attentions 不存在或不可靠，因此改用独立计算
        # 这里提供一个折中：不保存 attentions，重新由 conv1 计算一次注意力，
        # 但通过 checkpoint 避免存储。
        def conv3_with_tied_attention(x, edge_index, conv1, conv2):
            # 在重计算时重新生成 conv1 的注意力
            with torch.no_grad():
                # 用conv1 的当前参数计算注意力系数（不计算值，只取 alpha）
                # 由于 GATConv 的 attention 不公开接口，简单起见可以直接调用一次无梯度的 conv1，
                # 并提取其 attentions 属性。这仅用于获取 attention，代价较小。
                # 更精细的做法是单独提取 alpha 计算，为保持简洁，这里直接调用并丢弃输出。
                _ = conv1(x, edge_index)          # 会更新 conv1.attentions
                tied_attn = conv1.attentions
            # 利用 conv2 的权重完成 conv3 计算
            return self._conv3_forward(
                conv2.lin_src.weight, conv2.lin_dst.weight,
                x, edge_index, tied_attn, conv1.negative_slope
            )

        if self.checkpointing:
            h3 = checkpoint(conv3_with_tied_attention,
                            h2_new, edge_index, self.conv1, self.conv2,
                            use_reentrant=False)
        else:
            # 原风格：直接使用 self.conv3，但依赖 data 拷贝
            # 为保持行为一致，这里同样采用共享权重转置，避免 data 拷贝
            # 同步 conv2 的转置权重到 conv3（只在非 checkpoint 模式下直接操作）
            self.conv3.lin_src = lambda x: x @ self.conv2.lin_src.weight.t()
            self.conv3.lin_dst = lambda x: x @ self.conv2.lin_dst.weight.t()
            # 实际调用 conv3 的 forward 不能直接用，因为 lin_src 不是一个 Linear。
            # 因此非 checkpoint 模式也统一用 _conv3_forward。
            tied_attn = self.conv1.attentions
            h3 = self._conv3_forward(
                self.conv2.lin_src.weight, self.conv2.lin_dst.weight,
                h2_new, edge_index, tied_attn, self.conv1.negative_slope
            )
        h3 = F.elu(h3)

        # ---- conv4 (无注意力) ----
        h4 = self.conv4(h3, edge_index, attention=False)
        del h3

        # ---- 可选 MLP ----
        if self.use_mlp:
            h4 = h4 + self.mlp(h4)   # 残差

        return h2_new, h4


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




# class STACAME_minibatch_large(torch.nn.Module):
#     def __init__(self, hidden_dims):
#         super(STACAME_minibatch_large, self).__init__()

#         [in_dim, num_hidden, out_dim] = hidden_dims
#         self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)

#     def forward(self, features, adjs, mode="batch"):
#         if mode == "batch":
#             for i, (edge_index, e_id, size) in enumerate(adjs):
#                 # Extract target node features
#                 if i==0:
#                     h1 = F.elu(self.conv1(features, edge_index))
#                 #elif i == 1:
#                     h2 = self.conv2(h1, edge_index, attention=False)
#                     self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
#                     self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
#                     self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
#                     self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
#                 #elif i == 2:
#                     h3 = F.elu(self.conv3(h2, edge_index, attention=True,
#                                   tied_attention=self.conv1.attentions))
#                 #elif i == 3:
#                     h4 = self.conv4(h3, edge_index, attention=False)
#         else:
#             h1 = F.elu(self.conv1(features, adjs))
#             h2 = self.conv2(h1, adjs, attention=False)
#             self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
#             self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
#             self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
#             self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
#             h3 = F.elu(self.conv3(h2, adjs, attention=True,
#                                   tied_attention=self.conv1.attentions))
#             h4 = self.conv4(h3, adjs, attention=False)

#         return h2, h4  # F.log_softmax(x, dim=-1)
        
#     def inference(self, x_all, all_loader):
#         # This function will be called in test
#         for i in range(4):
#             xs = []
#             for batch_size, n_id, adj in all_loader:
#                 edge_index, _, size = adj.to(device)
#                 x = x_all[n_id].to(device)
#                 x_target = x[:size[1]]
#                 x = self.convs[i]((x, x_target), edge_index)
#                 if i != self.num_layers - 1:
#                     x = self.bns[i](x)
#                     x = F.relu(x)
#                     x = F.dropout(x, p=self.dropout, training=self.training)

#                 # Append the node embeddings to xs
#                 xs.append(x.cpu())

#             # Concat all embeddings into one tensor
#             x_all = torch.cat(xs, dim=0)



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



class STACAME_lightdecoder_minibatch(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STACAME_lightdecoder_minibatch, self).__init__()
        in_dim, num_hidden, out_dim = hidden_dims
        self.conv1 = GATConv(out_dim, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, adjs, mode="batch"):
        """
        features: 节点特征矩阵 (N, out_dim)
        adjs:     当 mode=="batch" 时，为子图邻接列表，通常长度为 1，
                  每个元素为 (edge_index, e_id, size)；
                  当 mode!="batch" 时，为完整图的 edge_index
        """
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
        """
        全图推理方法（可选），采用逐层计算所有节点嵌入的方式。
        subgraph_loader 为 PyG 的 DataLoader，通常设置 batch_size 较大，
        并按层顺序提供子图。
        """
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
