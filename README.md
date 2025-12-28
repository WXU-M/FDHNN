# FDHNN
<img width="10690" height="4351" alt="FDHNN" src="https://github.com/user-attachments/assets/d3544c83-503b-4664-8510-27a7773fce5c" />

#python 3.8  

#numpy 1.24.4

#pytorch 2.3.0

#pytorch-sparse 0.6.15

#pytorch-scatter 2.1.2

#pytorch-cluster 1.6.3

#pytorch-spline 1.2.2

#For graph datasets: 'Cora', 'Citeseer', 'PubMed', 'Photo', 'Computers','Chameleon', 'Squirrel'

python main.py --dataset 'Cora' --model 'dhl' --backbone 'gcn' --split_ratio -1

#For visual object dataset: '40','NTU'

python main.py --dataset '40' --model 'dhl' --backbone 'linear' --split_ratio 0.8
