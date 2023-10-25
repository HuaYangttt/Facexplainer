
# GCN: ba_2motifs, bace, bbbp, graph_sst2, mutag, twitter, redditbinary, collab
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=ba_2motifs models=gcn
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=bace models=gcn
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=bbbp models=gcn
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=graph_sst2 models=gcn
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=mutag models=gcn
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=twitter models=gcn
CUDA_VISIBLE_DEVICES=2 python train_gnns.py datasets=redditbinary models=gcn
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=collab models=gcn
# GIN: mutag
CUDA_VISIBLE_DEVICES=5 python train_gnns.py datasets=mutag models=gin
# GAT: graph_sst2
CUDA_VISIBLE_DEVICES=4 python train_gnns.py datasets=graph_sst2 models=gat











