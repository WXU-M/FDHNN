import math
from collections import defaultdict
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv


from sklearn.cluster import KMeans
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class HConstructorfor_graph(nn.Module):
    def __init__(self, in_features, out_features, num_classes, t,num_edges):
        super(HConstructorfor_graph, self).__init__()
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

    def ajust_edges(self, s_level, args):
        if args.stage != 'train':
            return

        if s_level > args.up_bound:
            self.num_edges = self.num_edges + 1
        elif s_level < args.low_bound:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges, args.min_num_edges)
        else:
            return

    def forward(self, edge_index, features,args):
        n_s = self.num_edges
        num_nodes = features.size(0)


        replicated_features = torch.cat([features for _ in range(self.t)], dim=0)


        transformed_features = [self.linear[i](features) for i in range(self.t)]
        transformed_features = torch.cat(transformed_features, dim=0)


        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        expanded_dst_nodes = dst_nodes.repeat(self.t)
        repeated_src_nodes = src_nodes.repeat_interleave(self.t)

        sim_scores = F.cosine_similarity(features[repeated_src_nodes], transformed_features[expanded_dst_nodes])


        sim_scores = sim_scores.view(-1, self.t)
        best_repl_idx = sim_scores.argmax(dim=1)


        best_repl_nodes = dst_nodes + best_repl_idx * num_nodes


        best_repl_edges = torch.stack([src_nodes, best_repl_nodes], dim=0)
        edge_index = torch.cat([edge_index, best_repl_edges], dim=1)


        extra_edges = []
        for i in range(self.t):
            edges = torch.stack([torch.arange(num_nodes, device=edge_index.device),
                                 torch.arange(num_nodes, device=edge_index.device) + i * num_nodes], dim=0)
            extra_edges.append(edges)

        extra_edges = torch.cat(extra_edges, dim=1)
        edge_index = torch.cat([edge_index, extra_edges], dim=1)


        all_features = torch.cat([features, transformed_features], dim=0)



        all_features2 = F.relu(all_features)
        all_features2 = F.dropout(all_features2, training=self.training)

        all_features2 = self.gcn_backbone[0](all_features2, edge_index)
        all_features2 = F.relu(all_features2)
        all_features2 = F.dropout(all_features2, training=self.training)
        all_features2 = self.gcn_backbone[1](all_features2, edge_index)

        all = F.relu(all_features2)
        all = F.dropout(all, training=self.training)
        all = self.linear1(all)


        H = torch.zeros(num_nodes, self.num_classes, device=all.device)
        for i in range(self.t + 1):
            classes = all[i * num_nodes:(i + 1) * num_nodes].argmax(dim=1)
            H[torch.arange(num_nodes), classes] += 1


        hyperedge_features = torch.zeros(self.num_classes, features.size(1), device=features.device)
        for j in range(self.num_classes):
            nodes_in_hyperedge = (H[:, j] > 0).nonzero(as_tuple=True)[0]
            if len(nodes_in_hyperedge) > 0:
                hyperedge_features[j] = all_features2[nodes_in_hyperedge].sum(dim=0)
        dots = torch.einsum('ni,ij->nj', all_features, hyperedge_features.T) * self.scale
        H = H.softmax(dim=0)

        return H, hyperedge_features, dots




class HConstructor_for_visual(nn.Module):
    def __init__(self, in_features, out_features, num_classes, t, num_edges):
        super(HConstructor_for_visual, self).__init__()
        self.scale = out_features ** -0.5
        self.num_edges = num_edges
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(t)])
        #self.linear = nn.ModuleList([KANLinear(in_features, out_features, "dog") for _ in range(t)])
        self.gcn = GCNConv(out_features, num_classes)
        self.num_classes = num_classes
        self.t = t
        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(out_features, num_classes))
        self.gcn_backbone.append(GCNConv(num_classes, num_classes))
        self.linear1 = nn.Linear(num_classes, self.num_edges)
        #self.linear1 = KANLinear(num_classes, self.num_edges, "dog")

        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(out_features, num_classes))
        self.linear_backbone.append(nn.Linear(num_classes, num_classes))
        # self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))




    def ajust_edges(self, s_level, args):
        if args.stage != 'train':
            return

        if s_level > args.up_bound:
            self.num_edges = self.num_edges + 1
        elif s_level < args.low_bound:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges, args.min_num_edges)
        else:
            return

    def forward(self, features, args):
        n_s = self.num_edges
        num_nodes = features.size(0)

        replicated_features = torch.cat([features for _ in range(self.t)], dim=0)


        transformed_features = [self.linear[i](features) for i in range(self.t)]
        transformed_features = torch.cat(transformed_features, dim=0)



        all_features = torch.cat([features, transformed_features], dim=0)
        #all_features = self.gcn(all_features, edge_index)

        all_features2 = F.relu(all_features)
        all_features2 = F.dropout(all_features2, training=self.training)

        all_features2 = self.linear_backbone[0](all_features2)
        all_features2 = F.relu(all_features2)
        all_features2 = F.dropout(all_features2, training=self.training)
        all_features2 = self.linear_backbone[1](all_features2)


        all = F.relu(all_features2)
        all = F.dropout(all, training=self.training)
        all = self.linear1(all)


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

        # cc = H.ceil().abs()
        # de = cc.sum(dim=0)
        # empty = (de == 0).sum()
        # s_level = 1 - empty / n_s
        #
        # self.ajust_edges(s_level, args)
        #
        # print("Num edges is: {}; Satuation level is: {}".format(self.num_edges, s_level))
        dots = torch.einsum('ni,ij->nj', all_features, hyperedge_features.T) * self.scale

        H = H.softmax(dim=0)
        return H, hyperedge_features, dots


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




class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, args, bias=True):
        super(HGNN_conv, self).__init__()

        self.num_node = args.num_node

        self.H2 = nn.ModuleList()
        self.H2.append(HConstructorfor_graph(in_ft, in_ft, 128,10,50))
        #self.H2.append(HConstructorfor_visual(in_ft, in_ft, 128, 10, 50))


        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()




        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.num_node, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))

        self.kan = KANLinear(in_ft, out_ft, "dog")


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,edge_index, x, args):# for graph
    #def forward(self, x, args): # for visual
        numnode = x.shape[0]
        H, edges, H_raw = self.H2[0](edge_index, x, args)# for graph
        #H, edges, H_raw = self.H2[0](x, args)# for visual

        num_nodes = torch.max(edge_index) + 1

        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)

        # 将 edge_index 中的边设置为 1
        adj_matrix[edge_index[0], edge_index[1]] = 1

        adj_matrix = self.mlp[0](adj_matrix)
        adj_matrix = F.relu(adj_matrix)

        adj_matrix = self.mlp[1](adj_matrix)
        adj_matrix = F.relu(adj_matrix)

        #edges = edges.matmul(self.weight)
        edges = self.kan(edges)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)


        #x = x + nodes #for Homophily

        x = torch.cat([x + nodes, adj_matrix], dim=1) #for Heterophily

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
        #self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim,hid_dim))



        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone.append(GCNConv(hid_dim, hid_dim))

        self.gcn_backbone2 = nn.ModuleList()
        self.gcn_backbone2.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone2.append(GCNConv(hid_dim, hid_dim))
        

        self.convs = nn.ModuleList()
        self.transfers = nn.ModuleList()

        for i in range(self.conv_number):
            self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges,args))
            self.transfers.append(nn.Linear(hid_dim*2, hid_dim)) #for Heterophily
            #self.transfers.append(nn.Linear(hid_dim, hid_dim)) #for Homophily


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
            #x = F.relu(self.linear_backbone[2](x))
            x = self.linear_backbone[2](x)
        elif args.backbone == 'gcn':
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x,edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x,edge_index)

        tmp = []
        H = []
        H_raw = []


        for i in range(self.conv_number):

            x, h, h_raw = self.convs[i](data['edge_index'],x,args)# for graph
            #x, h, h_raw = self.convs[i](x, args)# for visual
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if args.transfer == 1:
                x = self.transfers[i](x)
                x = F.relu(x)
            tmp.append(x)
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

        super(GCN, self).__init__()
        # graph convolution
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

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

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
