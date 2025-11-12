from utils import setup_seed, arg_parse, visualization
from load_data import load_data
from train import train_dhl,train_gcn,train_mlp,train_gat
import json
import torch
from networks import HGNN_classifier, GCN, GAT, MLP
import torch.nn.functional as F


chosse_trainer = {
    'dhl':train_dhl,
    'gcn':train_gcn,
    'MLP':train_mlp,
    'gat':train_gat
}

args = arg_parse()

setup_seed(args.seed)
data = load_data(args)

fts = data['fts']
lbls = data['lbls']

args.in_dim = fts.shape[1]
args.num_node = fts.shape[0]
args.out_dim = lbls.max().item() + 1
args.min_num_edges = args.k_e

args_list = []

best_acc = chosse_trainer[args.model](data, args)

args.best_acc = best_acc
args_list.append(args.__dict__)

############################################## visualization
chosse_model = {
    'dhl':HGNN_classifier,
    'gcn':GCN,
    'MLP':MLP,
    'gat':GAT
}

# model = chosse_model[args.model](args)
# state_dict = torch.load('model.pth',map_location=args.device)
# model.load_state_dict(state_dict)
# model.to(args.device)

# 先读 checkpoint
state_dict = torch.load('model.pth', map_location=args.device)

# 1) 从 checkpoint 里推断超边数 m，并写回 args
for k, v in state_dict.items():
    if k.endswith('linear1.weight'):        # 例如 convs.0.H2.0.linear1.weight
        args.num_edges = v.shape[0]         # 让初始化与 ckpt 的 m 对齐
        break

# 2) 用对齐后的 m 创建模型
model = chosse_model[args.model](args)

# 3) （可选）若构图器支持动态重建，显式同步一次
try:
    h = model.convs[0].H2[0]
    if hasattr(h, '_resize_output_layer'):
        h._resize_output_layer(args.num_edges)
except Exception:
    pass

# 4) 载入参数；strict=False 可兼容个别尺寸/缺失键
model.load_state_dict(state_dict, strict=False)
model.to(args.device)

args.stage = 'train'
model.eval()
mask = data['test_idx']
labels = data['lbls'][mask]

out, x, H, H_raw = model(data,args)
pred = F.log_softmax(out, dim=1)

_, pred = pred[mask].max(dim=1)
correct = int(pred.eq(labels).sum().item())
acc = correct / len(labels)

print("Acc ===============> ", acc)

visualization(model, data, args, title=None)
# with open('commandline_args{}.txt'.format(args.cuda), 'w') as f:
#     json.dump([args.__dict__,args.__dict__], f, indent=2)