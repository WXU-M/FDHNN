import math
from collections import defaultdict
#from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv


from sklearn.cluster import KMeans
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class HConstructor20(nn.Module):
    def __init__(self, in_features, out_features, num_classes, t,num_edges):
        super(HConstructor20, self).__init__()
        self.scale = out_features ** -0.5
        self.num_edges = num_edges
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(t)])

        # self.linear = nn.ModuleList([KANLinear(in_features, out_features, "dog") for _ in range(t)])
        self.gcn = GCNConv(out_features, num_classes)
        self.num_classes = num_classes
        self.t = t
        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(out_features, num_classes))
        self.gcn_backbone.append(GCNConv(num_classes, num_classes))
        self.linear1 = nn.Linear(num_classes, self.num_edges)
        # self.linear1 = KANLinear(num_classes, self.num_edges, "dog")

        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(out_features, num_classes))
        self.linear_backbone.append(nn.Linear(num_classes, num_classes))
        # self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(num_classes, num_classes))

    def _resize_output_layer(self, new_m):
        if new_m == self.linear1.out_features:
            return
        old = self.linear1
        in_f = old.in_features
        out_f = new_m
        device = next(old.parameters()).device
        new_linear = nn.Linear(in_f, out_f).to(device)

        # 拷贝重叠部分的权重和偏置
        keep = min(old.out_features, out_f)
        with torch.no_grad():
            new_linear.weight[:keep, :] = old.weight[:keep, :]
            new_linear.bias[:keep] = old.bias[:keep]
        self.linear1 = new_linear

    def ajust_edges(self, s_level, args):
        if args.stage != 'train':
            return
        new_m = self.num_edges
        if s_level > args.up_bound:
            new_m = self.num_edges + 1
        elif s_level < args.low_bound:
            new_m = max(self.num_edges - 1, args.min_num_edges)

        if new_m != self.num_edges:
            self.num_edges = new_m
            self._resize_output_layer(new_m)

    def forward(self, edge_index, features,args):
        n_s = self.num_edges
        num_nodes = features.size(0)

        # 1. 复制节点及其特征
        replicated_features = torch.cat([features for _ in range(self.t)], dim=0)

        # 2. 线性变换
        transformed_features = [self.linear[i](features) for i in range(self.t)]
        transformed_features = torch.cat(transformed_features, dim=0)




        # 4. 计算每个原件节点的邻居与所有副本节点的相似度，并连接到最相似的副本节点
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        expanded_dst_nodes = dst_nodes.repeat(self.t)
        repeated_src_nodes = src_nodes.repeat_interleave(self.t)

        # 检查生成的索引是否在范围内
        sim_scores = F.cosine_similarity(features[repeated_src_nodes], transformed_features[expanded_dst_nodes])

        # 重塑相似度矩阵以选择每个邻居与所有副本节点中的最相似者
        sim_scores = sim_scores.view(-1, self.t)
        best_repl_idx = sim_scores.argmax(dim=1)

        # 计算最相似的副本节点的索引
        best_repl_nodes = dst_nodes + best_repl_idx * num_nodes

        # 连接每个原件节点的邻居到最相似的副本节点
        best_repl_edges = torch.stack([src_nodes, best_repl_nodes], dim=0)
        edge_index = torch.cat([edge_index, best_repl_edges], dim=1)

        # 3. 连接每个原件节点和其所有副本节点
        extra_edges = []
        for i in range(self.t):
            edges = torch.stack([torch.arange(num_nodes, device=edge_index.device),
                                 torch.arange(num_nodes, device=edge_index.device) + i * num_nodes], dim=0)
            extra_edges.append(edges)

        extra_edges = torch.cat(extra_edges, dim=1)
        edge_index = torch.cat([edge_index, extra_edges], dim=1)

        # 5. 使用GCN进行节点分类
        all_features = torch.cat([features, transformed_features], dim=0)
        # all_features = self.gcn(all_features, edge_index)

        # all_features2 = F.relu(all_features)
        # all_features2 = F.dropout(all_features2, training=self.training)
        #
        # all_features2 = self.gcn(all_features2, edge_index)

        all_features2 = F.relu(all_features)
        all_features2 = F.dropout(all_features2, training=self.training)

        all_features2 = self.gcn_backbone[0](all_features2, edge_index)
        all_features2 = F.relu(all_features2)
        all_features2 = F.dropout(all_features2, training=self.training)
        all_features2 = self.gcn_backbone[1](all_features2, edge_index)

        all = F.relu(all_features2)
        all = F.dropout(all, training=self.training)
        all = self.linear1(all)

        # 6. 构造超图和超边邻接矩阵H
        #H = torch.zeros(num_nodes, self.num_classes, device=all.device)
        H = torch.zeros(num_nodes, n_s, device=all.device)
        for i in range(self.t + 1):
            classes = all[i * num_nodes:(i + 1) * num_nodes].argmax(dim=1)
            H[torch.arange(num_nodes), classes] += 1

        # 7. 计算超边特征矩阵
        hyperedge_features = torch.zeros(n_s, features.size(1), device=features.device)
        for j in range(n_s):
            nodes_in_hyperedge = (H[:, j] > 0).nonzero(as_tuple=True)[0]
            if len(nodes_in_hyperedge) > 0:
                hyperedge_features[j] = all_features2[nodes_in_hyperedge].sum(dim=0)

        dots = torch.einsum('ni,ij->nj', all_features, hyperedge_features.T) * self.scale
        #H = H.softmax(dim=0)

        cc = H.ceil().abs()  # 二值化
        de = cc.sum(dim=0)  # 每条超边的度
        empty = (de == 0).sum()
        s_level = 1 - empty.float() / n_s  # S_H = 1 - |E_empty|/|E|

        self.ajust_edges(s_level.item(), args)

        return H, hyperedge_features, dots
# class HConstructor10(nn.Module):
#     def __init__(self, in_features, out_features, num_classes, t, num_edges):
#         super(HConstructor10, self).__init__()
#         self.scale = out_features ** -0.5
#         self.num_edges = num_edges
#         self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(t)])
#         self.gcn = GCNConv(out_features, num_classes)
#         self.num_classes = num_classes
#         self.t = t
#         self.gcn_backbone = nn.ModuleList()
#         self.gcn_backbone.append(GCNConv(out_features, num_classes))
#         self.gcn_backbone.append(GCNConv(num_classes, num_classes))
#         self.linear1 = nn.Linear(num_classes, self.num_edges)
#
#     def _resize_output_layer(self, new_m):
#         if new_m == self.linear1.out_features:
#             return
#         old = self.linear1
#         in_f = old.in_features
#         out_f = new_m
#         device = next(old.parameters()).device
#         new_linear = nn.Linear(in_f, out_f).to(device)
#
#         # 拷贝重叠部分的权重和偏置
#         keep = min(old.out_features, out_f)
#         with torch.no_grad():
#             new_linear.weight[:keep, :] = old.weight[:keep, :]
#             new_linear.bias[:keep] = old.bias[:keep]
#         self.linear1 = new_linear
#
#     def ajust_edges(self, s_level, args):
#         if args.stage != 'train':
#             return
#         new_m = self.num_edges
#         if s_level > args.up_bound:
#             new_m = self.num_edges + 1
#         elif s_level < args.low_bound:
#             new_m = max(self.num_edges - 1, args.min_num_edges)
#
#         if new_m != self.num_edges:
#             self.num_edges = new_m
#             self._resize_output_layer(new_m)
#
#     def forward(self, edge_index, features, args):
#         n_s = self.num_edges
#         num_nodes = features.size(0)
#
#         # 1. 复制节点及其特征
#         replicated_features = torch.cat([features for _ in range(self.t)], dim=0)
#
#         # 2. 线性变换
#         transformed_features = [self.linear[i](features) for i in range(self.t)]
#         transformed_features = torch.cat(transformed_features, dim=0)
#
#         # 3. 计算相似度并调整边
#         src_nodes = edge_index[0]
#         dst_nodes = edge_index[1]
#
#         new_edges_list = []
#         for i in range(self.t):
#             sim_orig = F.cosine_similarity(transformed_features[src_nodes], features[dst_nodes])
#             sim_repl = F.cosine_similarity(transformed_features[src_nodes + i * num_nodes], features[dst_nodes])
#             new_edges = dst_nodes + (i + 1) * num_nodes
#             mask = sim_repl > sim_orig
#             edge_index_new = torch.cat([src_nodes.unsqueeze(0), new_edges.unsqueeze(0)], dim=0)[:, mask]
#             new_edges_list.append(edge_index_new)
#
#         edge_index = torch.cat([edge_index] + new_edges_list, dim=1)
#
#         # 4. 连接原始节点和复制节点
#         extra_edges = [torch.stack([torch.arange(num_nodes, device=edge_index.device), torch.arange(num_nodes, device=edge_index.device) + i * num_nodes], dim=0) for i in range(1, self.t + 1)]
#         extra_edges = torch.cat(extra_edges, dim=1)
#         edge_index = torch.cat([edge_index, extra_edges], dim=1)
#
#         # 5. 使用GCN进行节点分类
#         all_features = torch.cat([features, replicated_features], dim=0)
#         #all_features = self.gcn(all_features, edge_index)
#
#         all_features2 = F.relu(all_features)
#         all_features2 = F.dropout(all_features2, training=self.training)
#
#         all_features2 = self.gcn_backbone[0](all_features2, edge_index)
#         all_features2 = F.relu(all_features2)
#         all_features2 = F.dropout(all_features2, training=self.training)
#         all_features2 = self.gcn_backbone[1](all_features2, edge_index)
#
#         all = F.relu(all_features2)
#         all = F.dropout(all, training=self.training)
#         all = self.linear1(all)
#
#         # 6. 构造超图和超边邻接矩阵H
#         # H = torch.zeros(num_nodes, self.num_classes, device=all_features2.device)
#         H = torch.zeros(num_nodes, n_s, device=all.device)
#         for i in range(self.t + 1):
#             classes = all[i * num_nodes:(i + 1) * num_nodes].argmax(dim=1)
#             H[torch.arange(num_nodes), classes] += 1
#
#         # 7. 计算超边特征矩阵
#         # 7. 计算超边特征矩阵 —— 行数必须是 n_s（当前超边数），不能用 self.num_classes
#         hyperedge_features = torch.zeros(n_s, features.size(1), device=features.device)
#         for j in range(n_s):
#             nodes_in_hyperedge = (H[:, j] > 0).nonzero(as_tuple=True)[0]
#             if len(nodes_in_hyperedge) > 0:
#                 hyperedge_features[j] = all_features2[nodes_in_hyperedge].sum(dim=0)
#
#         dots = torch.einsum('ni,ij->nj', all_features, hyperedge_features.T) * self.scale
#
#         cc = H.ceil().abs()  # 二值化
#         de = cc.sum(dim=0)  # 每条超边的度
#         empty = (de == 0).sum()
#         s_level = 1 - empty.float() / n_s  # S_H = 1 - |E_empty|/|E|
#
#         self.ajust_edges(s_level.item(), args)
#
#         H = H.softmax(dim=0)
#         return H, hyperedge_features, dots

class HConstructor10(nn.Module):
    def __init__(self, in_features, out_features, num_classes, t, num_edges):
        super(HConstructor10, self).__init__()
        self.scale = out_features ** -0.5
        self.num_edges = num_edges
        self.num_classes = num_classes
        self.t = t

        # t 个线性变换
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(t)])

        # backbone：仿照你上一份代码，用 MLP 而不是 GCN
        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(out_features, num_classes))
        self.linear_backbone.append(nn.Linear(num_classes, num_classes))

        # 输出到超边的线性层（自适应调整输出维度）
        self.linear1 = nn.Linear(num_classes, self.num_edges)

    def _resize_output_layer(self, new_m):
        if new_m == self.linear1.out_features:
            return
        old = self.linear1
        in_f = old.in_features
        out_f = new_m
        device = next(old.parameters()).device
        new_linear = nn.Linear(in_f, out_f).to(device)

        # 拷贝重叠部分的权重和偏置
        keep = min(old.out_features, out_f)
        with torch.no_grad():
            new_linear.weight[:keep, :] = old.weight[:keep, :]
            new_linear.bias[:keep] = old.bias[:keep]
        self.linear1 = new_linear

    def ajust_edges(self, s_level, args):
        if args.stage != 'train':
            return
        new_m = self.num_edges
        if s_level > args.up_bound:
            new_m = self.num_edges + 1
        elif s_level < args.low_bound:
            new_m = max(self.num_edges - 1, args.min_num_edges)

        if new_m != self.num_edges:
            self.num_edges = new_m
            self._resize_output_layer(new_m)

    def forward(self, features, args):
        # 注意：这里不再接收 edge_index，只用 feature
        n_s = self.num_edges
        num_nodes = features.size(0)

        # 1. 复制节点及其特征（保留这一步，虽然后面主要用 transformed_features）
        replicated_features = torch.cat([features for _ in range(self.t)], dim=0)

        # 2. 线性变换（仿照你第一段代码的写法）
        transformed_features = [self.linear[i](features) for i in range(self.t)]
        transformed_features = torch.cat(transformed_features, dim=0)

        # 3. 不再用 edge_index 调整边，只根据特征来做后面的超边表示和相似度
        #    all_features 直接由原特征和线性变换后的特征拼起来
        all_features = torch.cat([features, transformed_features], dim=0)

        # 4. backbone（MLP）提取用于分配超边的表示
        all_features2 = F.relu(all_features)
        all_features2 = F.dropout(all_features2, training=self.training)

        all_features2 = self.linear_backbone[0](all_features2)
        all_features2 = F.relu(all_features2)
        all_features2 = F.dropout(all_features2, training=self.training)
        all_features2 = self.linear_backbone[1](all_features2)

        # 5. 通过 linear1 输出对每条超边的“logit”
        all = F.relu(all_features2)
        all = F.dropout(all, training=self.training)
        all = self.linear1(all)  # 维度：( (t+1)*num_nodes, n_s )

        # 6. 构造超图和超边邻接矩阵 H (num_nodes × n_s)
        device = all.device
        H = torch.zeros(num_nodes, n_s, device=device)
        for i in range(self.t + 1):
            classes = all[i * num_nodes:(i + 1) * num_nodes].argmax(dim=1)
            H[torch.arange(num_nodes, device=device), classes] += 1

        # 7. 计算超边特征矩阵 —— 行数是当前超边数 n_s
        hyperedge_features = torch.zeros(n_s, features.size(1), device=features.device)
        for j in range(n_s):
            nodes_in_hyperedge = (H[:, j] > 0).nonzero(as_tuple=True)[0]
            if len(nodes_in_hyperedge) > 0:
                hyperedge_features[j] = all_features2[nodes_in_hyperedge].sum(dim=0)

        # 8. 使用特征做相似度（节点–超边间的点积相似度）
        #    只依赖 all_features 和 hyperedge_features，不用 edge_index
        dots = torch.einsum('ni,ij->nj', all_features, hyperedge_features.T) * self.scale

        # 9. 计算饱和度，做自适应超边数调整（保留你的原始逻辑）
        cc = H.ceil().abs()        # 二值化
        de = cc.sum(dim=0)         # 每条超边的度
        empty = (de == 0).sum()
        s_level = 1 - empty.float() / n_s  # S_H = 1 - |E_empty|/|E|

        self.ajust_edges(s_level.item(), args)

        H = H.softmax(dim=0)
        return H, hyperedge_features, dots


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features,
                                                 in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            # You can try many more wavelet types ...
        elif self.wavelet_type == 'log':
            log = (x_scaled ** 2 - 2) * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = log
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'marr':
            marr = (x_scaled ** 2 - 2) * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = marr
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'gaussian_derivative':
            gauss_deriv = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = gauss_deriv
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight1)

        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        # Apply batch normalization
        return self.bn(combined_output)

class HConstructor9(nn.Module):
    def __init__(self, in_features, out_features, num_classes, t, num_edges):
        super(HConstructor9, self).__init__()
        self.scale = out_features ** -0.5
        self.num_edges = num_edges
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(t)])
        #self.linear = nn.ModuleList([KANLinear(in_features, out_features, "dog") for _ in range(t)])
        #self.gcn = GCNConv(out_features, num_classes)
        self.num_classes = num_classes
        self.t = t
        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(out_features, num_classes))
        self.gcn_backbone.append(GCNConv(num_classes, num_classes))
        self.linear1 = nn.Linear(num_classes, self.num_edges)
        #self.linear1 = KANLinear(num_classes, self.num_edges, "dog")

        # self.mlp = nn.Sequential(
        #     nn.Linear(out_features, num_classes),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_classes, num_classes),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_classes, num_classes),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_classes, num_classes)
        # )

    def _resize_output_layer(self, new_m):
        if new_m == self.linear1.out_features:
            return
        old = self.linear1
        in_f = old.in_features
        out_f = new_m
        device = next(old.parameters()).device
        new_linear = nn.Linear(in_f, out_f).to(device)

        # 拷贝重叠部分的权重和偏置
        keep = min(old.out_features, out_f)
        with torch.no_grad():
            new_linear.weight[:keep, :] = old.weight[:keep, :]
            new_linear.bias[:keep] = old.bias[:keep]
        self.linear1 = new_linear

    def ajust_edges(self, s_level, args):
        if args.stage != 'train':
            return
        new_m = self.num_edges
        if s_level > args.up_bound:
            new_m = self.num_edges + 1
        elif s_level < args.low_bound:
            new_m = max(self.num_edges - 1, args.min_num_edges)

        if new_m != self.num_edges:
            self.num_edges = new_m
            self._resize_output_layer(new_m)

    def forward(self, edge_index, features, args):
        n_s = self.num_edges
        num_nodes = features.size(0)

        # 1. 复制节点及其特征
        replicated_features = torch.cat([features for _ in range(self.t)], dim=0)

        # 2. 线性变换
        transformed_features = [self.linear[i](features) for i in range(self.t)]
        transformed_features = torch.cat(transformed_features, dim=0)

        # 3. 计算相似度并调整边
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        new_edges_list = []
        for i in range(self.t):
            sim_orig = F.cosine_similarity(features[src_nodes], features[dst_nodes])
            sim_repl = F.cosine_similarity(transformed_features[src_nodes + i * num_nodes], features[dst_nodes])
            new_edges = dst_nodes + (i + 1) * num_nodes
            mask = sim_repl > sim_orig
            edge_index_new = torch.cat([src_nodes.unsqueeze(0), new_edges.unsqueeze(0)], dim=0)[:, mask]
            new_edges_list.append(edge_index_new)

        edge_index = torch.cat([edge_index] + new_edges_list, dim=1)

        # 4. 连接原始节点和复制节点
        extra_edges = [torch.stack([torch.arange(num_nodes, device=edge_index.device), torch.arange(num_nodes, device=edge_index.device) + i * num_nodes], dim=0) for i in range(1, self.t + 1)]
        extra_edges = torch.cat(extra_edges, dim=1)
        edge_index = torch.cat([edge_index, extra_edges], dim=1)

        # 5. 使用GCN进行节点分类
        all_features = torch.cat([features, replicated_features], dim=0)
        #all_features = self.gcn(all_features, edge_index)

        all_features2 = F.relu(all_features)
        all_features2 = F.dropout(all_features2, training=self.training)

        all_features2 = self.gcn_backbone[0](all_features2, edge_index)
        all_features2 = F.relu(all_features2)
        all_features2 = F.dropout(all_features2, training=self.training)
        all_features2 = self.gcn_backbone[1](all_features2, edge_index)

        #all_features2 = self.mlp(all_features2)

        all = F.relu(all_features2)
        all = F.dropout(all, training=self.training)
        all = self.linear1(all)

        # 6. 构造超图和超边邻接矩阵H
        # H = torch.zeros(num_nodes, self.num_classes, device=all_features2.device)
        H = torch.zeros(num_nodes, n_s, device=all.device)
        for i in range(self.t + 1):
            classes = all[i * num_nodes:(i + 1) * num_nodes].argmax(dim=1)
            H[torch.arange(num_nodes), classes] += 1

        # 7. 计算超边特征矩阵
        hyperedge_features = torch.zeros(n_s, features.size(1), device=features.device)
        for j in range(n_s):
            nodes_in_hyperedge = (H[:, j] > 0).nonzero(as_tuple=True)[0]
            if len(nodes_in_hyperedge) > 0:
                hyperedge_features[j] = all_features2[nodes_in_hyperedge].sum(dim=0)

        cc = H.ceil().abs()  # 二值化
        de = cc.sum(dim=0)  # 每条超边的度
        empty = (de == 0).sum()
        s_level = 1 - empty.float() / n_s  # S_H = 1 - |E_empty|/|E|

        self.ajust_edges(s_level.item(), args)

        dots = torch.einsum('ni,ij->nj', all_features, hyperedge_features.T) * self.scale

        H = H.softmax(dim=0)
        return H, hyperedge_features, dots




class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, args, bias=True):
        super(HGNN_conv, self).__init__()

        self.num_node = args.num_node

        #self.HConstructor = HConstructor(num_edges, in_ft)

        self.H2 = nn.ModuleList()
        name = args.dataset
        if name in {'Cora', 'Citeseer', 'PubMed','Photo', 'Computers'}:
            self.H2.append(HConstructor9(in_ft, in_ft, in_ft, args.k, args.num_edges))
        elif name in {'Chameleon', 'Squirrel'}:
            self.H2.append(HConstructor20(in_ft, in_ft, in_ft, args.k, args.num_edges))
        else:
            self.H2.append(HConstructor10(in_ft, in_ft, in_ft, args.k, args.num_edges))

        #self.H2.append(HConstructor20(in_ft, in_ft, in_ft,15,100))
        self.d_k =out_ft


        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.num_node, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))

        self.W_Q = nn.Parameter(torch.randn(self.num_node, out_ft))  # Query 权重
        self.W_K = nn.Parameter(torch.randn(self.num_node, out_ft))  # Key 权重
        self.W_V = nn.Parameter(torch.randn(self.num_node, out_ft))  # Value 权重

        # 初始化权重（可选）
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)


        # self.mlp.append(KANLinear(self.num_node, out_ft, "dog"))
        # self.mlp.append(KANLinear(out_ft, out_ft, "dog"))

        self.kan = KANLinear(in_ft, out_ft, "dog")

        #self.norm_input = nn.LayerNorm(in_ft)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def self_attention_on_adj(self, A):
        """
        对邻接矩阵 A 应用自注意力机制。
        :param A: 输入邻接矩阵 (N x N)
        :return: 输出特征矩阵 (N x d_k)
        """
        # 计算 Q, K, V
        Q = torch.matmul(A, self.W_Q)  # (N x d_k)
        K = torch.matmul(A, self.W_K)  # (N x d_k)
        V = torch.matmul(A, self.W_V)  # (N x d_k)

        # 计算注意力分数
        S = torch.matmul(Q, K.transpose(0, 1)) / (self.d_k ** 0.5)  # (N x N)
        attention = F.softmax(S, dim=-1)  # 对每行做 softmax

        # 加权聚合 Value
        output = torch.matmul(attention, V)  # (N x d_k)
        return output

    def forward(self, A):
        """
        前向传播，调用自注意力模块。
        :param A: 输入邻接矩阵 (N x N)
        :return: 输出特征矩阵 (N x d_k)
        """
        output = self.self_attention_on_adj(A)
        return output

    def forward(self,edge_index, x, args):
        numnode = x.shape[0]
        #x = self.norm_input(x)

        #edges, H, H_raw = self.HConstructor(x, args)
        H, edges, H_raw = self.H2[0](edge_index, x, args)
        #H_raw = edges
        #H = H[0:numnode,:]+H[numnode:,:]
        #H_raw = H_raw[0:numnode, :] + H_raw[numnode:, :]
        #H,edges = self.H2[0](edge_index, x)

        num_nodes = torch.max(edge_index) + 1

        # 创建 n x n 的邻接矩阵并初始化为零
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)

        # 将 edge_index 中的边设置为 1
        adj_matrix[edge_index[0], edge_index[1]] = 1

        adj_matrix =self.self_attention_on_adj(adj_matrix)

        #adj_matrix = self.mlp[0](adj_matrix)
        #adj_matrix =F.relu(adj_matrix)

        #adj_matrix = self.mlp[1](adj_matrix)
        #adj_matrix = F.relu(adj_matrix)

        #edges = edges.matmul(self.weight)
        # -----------------------------------------------

        edges = self.kan(edges)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        #x = self.mlp[0](x) + self.mlp[1](nodes)
        #x = self.mlp[0](torch.cat([x,nodes],dim=1))
        #x = x + nodes

        #x = self.kan(x)

        name = args.dataset
        if name in {'Cora', 'Citeseer', 'PubMed','Photo', 'Computers'}:
            x = x + nodes
        elif name in {'Chameleon', 'Squirrel'}:
            x =  torch.cat([x+nodes,adj_matrix],dim=1)
        return x, H, H_raw

class HGNN_conv_noedge(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, args, bias=True):
        super(HGNN_conv_noedge, self).__init__()

        #self.HConstructor = HConstructor(num_edges, in_ft)

        self.H2 = nn.ModuleList()

        self.H2.append(HConstructor10(in_ft, in_ft, in_ft, args.k, args.num_edges))


        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_ft, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))
        #self.mlp.append(KANLinear(in_ft, out_ft, "dog"))
        #self.mlp.append(KANLinear(out_ft, out_ft, "dog"))

        self.kan = KANLinear(in_ft, out_ft, "dog")


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # def forward(self,edge_index, x, args):
    def forward(self, x, args):
        numnode = x.shape[0]

        # H, edges, H_raw = self.H2[0](edge_index, x, args)
        H, edges, H_raw = self.H2[0](x, args)
        #H_raw = edges
        #H = H[0:numnode,:]+H[numnode:,:]
        #H_raw = H_raw[0:numnode, :] + H_raw[numnode:, :]
        #H,edges = self.H2[0](edge_index, x)
        #edges = edges.matmul(self.weight)
        edges = self.kan(edges)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        #x = self.mlp[0](x) + self.mlp[1](nodes)
        x = x + nodes
        return x, H, H_raw




class HGNN_classifier(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(HGNN_classifier, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim
        num_edges = args.num_edges
        self.conv_number = args.conv_number

        self.dropout = dropout

        #self.linear_backbone = nn.Linear(in_dim,hid_dim)


        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(in_dim,hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim,hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim,hid_dim))



        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone.append(GCNConv(hid_dim, hid_dim))



        self.convs = nn.ModuleList()
        self.transfers = nn.ModuleList()

        name = args.dataset
        if name in {'Chameleon', 'Squirrel'}:
            for i in range(self.conv_number):
                self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges, args))
                self.transfers.append(nn.Linear(hid_dim * 2, hid_dim))
        elif name in {'40','NTU'}:
            for i in range(self.conv_number):

                self.convs.append(HGNN_conv_noedge(hid_dim, hid_dim, num_edges,args))
                self.transfers.append(nn.Linear(hid_dim, hid_dim))
        else:
            for i in range(self.conv_number):
                self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges, args))
                self.transfers.append(nn.Linear(hid_dim, hid_dim))



        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_number * hid_dim, out_dim),
        )

    def forward(self, data, args):

        if args.backbone == 'linear':
            x = data['fts']
            #x = self.linear_backbone[0](x)
            x = F.relu(self.linear_backbone[0](x))
            x = F.relu(self.linear_backbone[1](x))
            x = self.linear_backbone[2](x)
        elif args.backbone == 'gcn':
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x,edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x,edge_index)

            # x2 = data['fts']
            # edge_index = data['edge_index']
            # x2 = self.gcn_backbone[0](x2, edge_index)
            # x2 = F.relu(x2)
            # x2 = F.dropout(x2, training=self.training)
            # x2 = self.gcn_backbone[1](x2, edge_index)

        tmp = []
        H = []
        H_raw = []


        for i in range(self.conv_number):
            #x = self.H2[0](data['edge_index'], data['fts'])
            if args.dataset == "40" or args.dataset == "NTU":
                x, h, h_raw = self.convs[i](x, args)
            else:
                x, h, h_raw = self.convs[i](data['edge_index'], x, args)

            #x1, h, h_raw = self.convs[i](data['edge_index'],x,args)
            x1 = F.relu(x)
            x1 = F.dropout(x1, training=self.training)
            if args.transfer == 1:
                x1 = self.transfers[i](x1)
                #x1 = self.transfers[i+1](x1)
                x1 = F.relu(x1)
            tmp.append(x1)
            H.append(h)
            H_raw.append(h_raw)



        x = torch.cat(tmp,dim=1)

        out = self.classifier(x)
        return out, x, H, H_raw

class GCN(nn.Module):
    def __init__(self, args, layer_number=2):

        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim
        self.num_node = args.num_node




        super(GCN, self).__init__()
        # graph convolution
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.num_node, 128))
        self.mlp.append(nn.Linear(128, 128))

        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GCNConv(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']

        num_nodes = torch.max(edge_index) + 1

        # 创建 n x n 的邻接矩阵并初始化为零
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)

        # 将 edge_index 中的边设置为 1
        adj_matrix[edge_index[0], edge_index[1]] = 1

        adj_matrix = self.mlp[0](adj_matrix)
        adj_matrix = F.relu(adj_matrix)

        adj_matrix = self.mlp[1](adj_matrix)
        adj_matrix = F.relu(adj_matrix)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        # x = torch.cat([x, adj_matrix], dim=1)


        out = self.classifier(x)
        return out, x, None, None

class GAT(nn.Module):
    def __init__(self, args, layer_number=2):
        super(GAT, self).__init__()
        
        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim

        # graph convolution
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GATConv(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        out = self.classifier(x)
        return out, x, None, None

class MLP(nn.Module):
    def __init__(self, args, dropout=0.5, bias=True):

        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim

        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hid_dim, out_dim)
        )
  

    def forward(self,data,args):
        x = data['fts']           

        out = self.mlp(x)            

        return out, None, None, None
