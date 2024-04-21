import os
import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid, CoraFull
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(z, decoder, pos_edge_index, neg_edge_index, K=20):
    y_pred_pos = decoder(z, pos_edge_index).cpu()
    y_pred_neg = decoder(z, neg_edge_index).cpu()

    y_pred = torch.cat([y_pred_pos, y_pred_neg], dim=0)
    y_true = torch.cat([torch.ones_like(y_pred_pos),
                        torch.zeros_like(y_pred_neg)], dim=0)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = (y_pred_pos > kth_score_in_negative_edges).sum() / y_pred_pos.shape[0]

    return float(auc), float(ap), float(hitsK)


def load_data(dataset, val_ratio=0.05, test_ratio=0.15):
    transform = T.NormalizeFeatures()
    path = '/home/data'
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(path, dataset, transform=transform)[0]
    elif dataset == 'CoraFull':
        data = CoraFull(f'{path}/{dataset}', transform=transform)[0]

    return train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
