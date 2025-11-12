import os.path as osp
import torch
from numpy import ndarray
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from torch import LongTensor
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, WikipediaNetwork, Actor
import scipy.io as scio
import numpy as np
import torchvision
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pickle
import itertools
import random
import sys








def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits(labels, num_classes, percls_trn=20, val_size=500, test_size=1000):
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(rest_index[:val_size], size=num_nodes)
    test_mask = index_to_mask(rest_index[val_size:val_size + test_size], size=num_nodes)

    return train_mask, val_mask, test_mask

def get_mask(y, train_ratio=0.6, test_ratio=0.2, device=None):
    if device is None:
        device = torch.device("cpu")
    train_indexes = list()
    test_indexes = list()
    val_indexes = list()
    npy = y.cpu().numpy()

    def get_sub_mask(sub_x_indexes):
        np.random.shuffle(sub_x_indexes)
        sub_train_count = int(len(sub_x_indexes) * train_ratio)
        sub_test_count = int(len(sub_x_indexes) * test_ratio)
        sub_train_indexes = sub_x_indexes[0:sub_train_count]
        sub_test_indexes = sub_x_indexes[sub_train_count:sub_train_count + sub_test_count]
        sub_val_indexes = sub_x_indexes[sub_train_count + sub_test_count:]
        return sub_train_indexes, sub_test_indexes, sub_val_indexes

    def flatten_np_list(np_list):
        total_size = sum([len(item) for item in np_list])
        result = ndarray(shape=total_size)
        last_i = 0
        for item in np_list:
            result[last_i:last_i + len(item)] = item
            last_i += len(item)
        return np.sort(result)

    for class_id in np.unique(npy):
        indexes = np.argwhere(npy == class_id).flatten().astype(int)
        m, n, q = get_sub_mask(indexes)
        train_indexes.append(m)
        test_indexes.append(n)
        val_indexes.append(q)
    train_indexes = LongTensor(flatten_np_list(train_indexes)).to(device)
    test_indexes = LongTensor(flatten_np_list(test_indexes)).to(device)
    val_indexes = LongTensor(flatten_np_list(val_indexes)).to(device)
    return train_indexes, test_indexes, val_indexes



def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], "__iter__"):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix

def sample_per_class(
    random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """
    Used in get_train_val_test_split, when we try to get a fixed number of examples per class
    """

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )

def get_train_val_test_split(
    random_state,
    labels,
    train_examples_per_class=None,
    val_examples_per_class=None,
    test_examples_per_class=None,
    train_size=None,
    val_size=None,
    test_size=None,
):

    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
            len(np.concatenate((train_indices, val_indices, test_indices)))
            == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def rand_train_test_idx(label, train_prop=.6, valid_prop=.2, ignore_negative=True, balance=False):
	""" Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
	""" randomly splits label into train/valid/test splits """
	if not balance:
		if ignore_negative:
			labeled_nodes = torch.where(label != -1)[0]
		else:
			labeled_nodes = label

		n = labeled_nodes.shape[0]
		train_num = int(n * train_prop)
		valid_num = int(n * valid_prop)

		perm = torch.as_tensor(np.random.permutation(n))

		train_indices = perm[:train_num]
		val_indices = perm[train_num:train_num + valid_num]
		test_indices = perm[train_num + valid_num:]

		if not ignore_negative:
			return train_indices, val_indices, test_indices

		train_idx = labeled_nodes[train_indices]
		valid_idx = labeled_nodes[val_indices]
		test_idx = labeled_nodes[test_indices]

		split_idx = {'train': train_idx,
					 'valid': valid_idx,
					 'test': test_idx}
	else:
		#         ipdb.set_trace()
		indices = []
		for i in range(label.max()+1):
			index = torch.where((label == i))[0].view(-1)
			index = index[torch.randperm(index.size(0))]
			indices.append(index)

		percls_trn = int(train_prop/(label.max()+1)*len(label))
		val_lb = int(valid_prop*len(label))
		train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
		rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
		rest_index = rest_index[torch.randperm(rest_index.size(0))]
		valid_idx = rest_index[:val_lb]
		test_idx = rest_index[val_lb:]
		split_idx = {'train': train_idx,
					 'valid': valid_idx,
					 'test': test_idx}
	return split_idx

def load_data2(args):
    """
    parses the dataset
    """
    dataset = args.dataset
    splits = args.splits
    device = torch.device(args.device)
    path = osp.abspath(__file__)         #当前文件绝对路径
    d_path = osp.dirname(path)           #当前文件所在目录
    # f_path = osp.dirname(d_path)         #当前文件所在目录的父目录
    f_path = osp.join(d_path, ('data2'))
    
    d_path_dict = {
        'ca_cora':osp.join(osp.join(f_path, ('coauthorship')),'cora'),
        'ca_dblp':osp.join(osp.join(f_path, ('coauthorship')),'dblp'),
        'cc_cora':osp.join(osp.join(f_path, ('cocitation')),'cora'),
        'cc_citeseer':osp.join(osp.join(f_path, ('cocitation')),'citeseer'),
        'ca_pubmed':osp.join(osp.join(f_path, ('cocitation')),'pubmed')
    }

    pickle_file = osp.join(d_path_dict[dataset], "splits", str(splits) + ".pickle")

    with open(osp.join(d_path_dict[dataset], 'features.pickle'), 'rb') as handle:
        features = pickle.load(handle).todense()

    with open(osp.join(d_path_dict[dataset], 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(pickle_file, 'rb') as H: 
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    with open(osp.join(d_path_dict[dataset], 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

    tmp_edge_index = []
    for key in hypergraph.keys():
        ms = hypergraph[key]
        tmp_edge_index.extend(list(itertools.permutations(ms,2)))
    
    edge_s = [ x[0] for x in tmp_edge_index]
    edge_e = [ x[1] for x in tmp_edge_index]

    edge_index = torch.LongTensor([edge_s,edge_e])

    features = torch.Tensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    data = {
        'fts':features,
        'edge_index':edge_index,
        'lbls':labels,
        'train_idx':train,
        'test_idx':test
    }

    return data


def load_cite(args):
    dname = args.dataset
    device = torch.device(args.device)
    path = osp.abspath(__file__)         #当前文件绝对路径
    d_path = osp.dirname(path)           #当前文件所在目录
    # f_path = osp.dirname(d_path)         #当前文件所在目录的父目录
    f_path = osp.join(d_path, ('data'))

    # 根据数据集名称选择对应的数据集类
    name = dname



    if name in {'Cora', 'Citeseer', 'PubMed'}:
        dataset = Planetoid(f_path, dname)
    elif name in {'Photo', 'Computers'}:
        dataset = Amazon(f_path, dname)
    elif name in {'Chameleon', 'Squirrel'}:
        dataset = WikipediaNetwork(f_path, dname)
    elif name.lower() == 'actor':
        dataset = Actor(f_path)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # #dataset = Planetoid(f_path,dname)      #dataset
    # dataset = Amazon(f_path, dname)
    # #dataset = WikipediaNetwork(f_path, dname)
    # #dataset = Actor(f_path, dname)




    tmp = dataset[0].to(device)
    fts = tmp.x
    lbls = tmp.y

    if args.split_ratio < 0:
        # train_idx = tmp.train_mask
        # val_idx = tmp.val_mask
        # test_idx = tmp.test_mask
        #------------------------------------

        # split_idx = rand_train_test_idx(lbls)
        # train_idx = split_idx['train']
        # val_idx = split_idx['valid']
        # test_idx = split_idx['test']
        #----------------------------------------

        # labels = binarize_labels(lbls.tolist())
        #
        # random_state = np.random.RandomState(0)
        # train_idx, val_idx, test_idx = get_train_val_test_split(
        #     random_state, labels, 20, 30
        # )
        #
        # train_idx = torch.tensor(train_idx)
        # val_idx = torch.tensor(val_idx)
        # test_idx = torch.tensor(test_idx)
        #----------------------------------------------

        train_idx, test_idx, val_idx = get_mask(lbls, 0.6, 0.2, device=device)
        #-------------------------------------------------------

        # train_idx, test_idx, val_idx = random_splits(lbls,7)


    else:
        nums = lbls.shape[0]
        num_train = int(nums * args.split_ratio)

        idx_list = [i for i in range(nums)]


        train_idx = random.sample(idx_list, num_train)
        test_idx = [i for i in idx_list if i not in train_idx]

        train_idx = torch.tensor(train_idx)
        test_idx = torch.tensor(test_idx)

        # num_train = int(nums * 0.5)
        # num_val = int(nums * 0.25)
        # num_test = nums - num_train - num_val

        # # 生成索引列表
        # idx_list = [i for i in range(nums)]
        #
        # # 随机打乱索引列表
        # random.shuffle(idx_list)
        #
        # # 获取训练集、验证集和测试集的索引
        # train_idx = idx_list[:num_train]
        # val_idx = idx_list[num_train:num_train + num_val]
        # test_idx = idx_list[num_train + num_val:]
        #
        # # 转换为 tensor
        # train_idx = torch.tensor(train_idx)
        # val_idx = torch.tensor(val_idx)
        # test_idx = torch.tensor(test_idx)

    data = {
        'fts':fts,
        'edge_index':tmp.edge_index,
        'lbls':lbls,
        'train_idx':train_idx,
        'val_idx':val_idx,
        'test_idx':test_idx
    }

    return data

def load_ft(args):
    if args.dataset == '40':
        data_dir = './data/ModelNet40_mvcnn_gvcnn.mat'
    elif args.dataset == 'NTU':
        data_dir = './data/NTU2012_mvcnn_gvcnn.mat'

    device = torch.device(args.device)
    feature_name = args.fts

    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32)
        fts = torch.Tensor(fts).to(device)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
        fts = torch.Tensor(fts).to(device)
    else:
        fts1 = data['X'][0].item().astype(np.float32)
        fts2 = data['X'][1].item().astype(np.float32)
        fts1 = torch.Tensor(fts1).to(device)
        fts2 = torch.Tensor(fts2).to(device)

        fts = torch.cat((fts1,fts2),dim=-1)

    if args.split_ratio < 0:
        train_idx = np.where(idx == 1)[0]
        test_idx = np.where(idx == 0)[0]
    else:
        nums = lbls.shape[0]
        num_train = int(nums * args.split_ratio)
        idx_list = [i for i in range(nums)]

        train_idx = random.sample(idx_list, num_train)
        test_idx = [i for i in idx_list if i not in train_idx]

    # train_idx = np.where(idx == 1)[0]
    # test_idx = np.where(idx == 0)[0]

    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    train_idx = torch.Tensor(train_idx).long().to(device)
    test_idx = torch.Tensor(test_idx).long().to(device)

    data = {
        'fts':fts,
        'lbls':lbls,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data

def load_data(args):
    if args.dataset in ['40','NTU']:
        return load_ft(args)
    elif args.dataset in ['Cora','Citeseer','PubMed','Photo','Computers','Chameleon','Squirrel','actor']:
        return load_cite(args)
    elif args.dataset in ['MINIST']:
        return load_minist(args)
    elif args.dataset in ['cora']:
        return load_citation_data()

def load_minist(args):
    device = torch.device(args.device)
    dataset = torchvision.datasets.MNIST(root='./data',transform=lambda x:list(x.getdata()),download=True)
    features = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]
    features = torch.Tensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    train_idx = [i for i in range(50000)]
    test_idx = [i for i in range(50000,60000)]

    data = {
        'fts':features,
        'lbls':labels,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_citation_data():
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    cfg = {
        'citation_root':'./data/gcn',
        'activate_dataset':'cora',
        'add_self_loop': True
    }


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))
    test_idx_range = np.sort(test_idx_reorder)

    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()

    G = nx.from_dict_of_lists(graph)
    # print("=====> ", G)
    # edge_list = G.adjacency_list()
    adjacency = G.adjacency()
    edge_list = []
    for item in adjacency:
        # print(list(item[1].keys()))
        edge_list.append(list(item[1].keys()))

    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])
    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'citeseer':
        n_category += 1                                         # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
            except ValueError:                              # labels[i] all zeros
                lbls[i] = n_category + 1                        # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    

    features = torch.Tensor(features)
    lbls = torch.LongTensor(lbls)

    data = {
        'fts':features,
        'lbls':lbls,
        'train_idx':idx_val,
        'test_idx':idx_test
    }

    return data

    # return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list