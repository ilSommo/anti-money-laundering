__version__ = '1.0.0'
__author__ = 'Martino Pulici'


import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from aml.load_data import load_data
from aml.models import GAT, GAT_MLP, MLP, MLP_GAT_MLP, SLP
from aml.utils import best_f1, evaluate


# Perform testing
def test():
    model.eval()
    output = model(features, adj)
    _, threshold = best_f1(output[idx_val], labels[idx_val])
    loss_test = loss(output[idx_test], labels[idx_test].unsqueeze(1))
    acc_test, prec_test, rec_test, f1_test = evaluate(
        output[idx_test], labels[idx_test], threshold=threshold)
    lr_auc = roc_auc_score(
        labels[idx_test].data.tolist(),
        output[idx_test].data.tolist())
    print(file,
          'loss = {:.3f},'.format(loss_test.data.item()),
          'accuracy = {:.3f},'.format(acc_test),
          'precision = {:.3f},'.format(prec_test),
          'recall = {:.3f},'.format(rec_test),
          'f1-score = {:.3f},'.format(f1_test),
          'ROC-AUC = {:.3f}'.format(lr_auc))
    lr_fpr, lr_tpr, _ = roc_curve(
        labels[idx_test].data.tolist(), output[idx_test].data.tolist())
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    return pd.concat([df,
                      pd.DataFrame([{'model_name': model_name,
                                     'lr': lr,
                                     'weight_decay': weight_decay,
                                     'hidden': hidden,
                                     'nb_heads': nb_heads,
                                     'dropout': dropout,
                                     'alpha': alpha,
                                     'loss': loss_test.data.item(),
                                     'accuracy': acc_test,
                                     'precision': prec_test,
                                     'recall': rec_test,
                                     'f1-score': f1_test,
                                     'ROC-AUC': lr_auc}])],
                     ignore_index=True)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder',
    type=str,
    default='checkpoints',
    help='.pkl files folder.')
parser.add_argument('--seed', type=int, default=28, help='Random seed.')
args = parser.parse_args()

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Initialize loss
loss = nn.BCELoss()

# Initialize dataframe
df = pd.DataFrame()

# Perform testing loop
for file in sorted(glob.glob(args.folder + '/*')):
    model_name, lr, weight_decay, hidden, nb_heads, dropout, alpha = os.path.basename(file)[
        :-4].split('_')
    lr = float(lr)
    weight_decay = float(weight_decay)
    hidden = int(hidden)
    nb_heads = int(nb_heads)
    dropout = float(dropout)
    alpha = float(alpha)
    if model_name == 'SLP':
        model = SLP(nfeat=features.shape[1], dropout=dropout)
    if model_name == 'MLP':
        model = MLP(nfeat=features.shape[1], dropout=dropout)
    if model_name == 'GAT':
        model = GAT(
            nfeat=features.shape[1],
            nhid=hidden,
            dropout=dropout,
            alpha=alpha,
            nheads=nb_heads)
    if model_name == 'GATMLP':
        model = GAT_MLP(
            nfeat=features.shape[1],
            nhid=hidden,
            dropout=dropout,
            alpha=alpha,
            nheads=nb_heads)
    if model_name == 'MLPGATMLP':
        model = MLP_GAT_MLP(
            nfeat=features.shape[1],
            nhid=hidden,
            dropout=dropout,
            alpha=alpha,
            nheads=nb_heads)
    if torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(file))
    else:
        model.load_state_dict(
            torch.load(
                file,
                map_location=torch.device('cpu')))
    df = test()

# Save test results
df.to_csv('test.csv', index=False)

# Plot ROC curves
ns_probs = [0 for _ in range(len(labels[idx_test].data.tolist()))]
ns_fpr, ns_tpr, _ = roc_curve(labels[idx_test].data.tolist(), ns_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('test.png')
