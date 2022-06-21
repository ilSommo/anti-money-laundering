from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Martino Pulici'


import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from aml.load_data import load_data
from aml.models import GAT, GAT_MLP, MLP, MLP_GAT_MLP, SLP
from aml.utils import best_f1, evaluate


# Perform evaluation
def eval():
    model.eval()
    output = model(features, adj)
    loss_val = loss(output[idx_val], labels[idx_val].unsqueeze(1))
    acc_val, prec_val, rec_val, f1_val = evaluate(
        output[idx_val], labels[idx_val])
    best_f1_val, best_f1_threshold = best_f1(output[idx_val], labels[idx_val])
    print(file,
          'loss = {:.3f},'.format(loss_val.data.item()),
          'accuracy = {:.3f},'.format(acc_val),
          'precision = {:.3f},'.format(prec_val),
          'recall = {:.3f},'.format(rec_val),
          'f1-score = {:.3f},'.format(f1_val),
          'best f1 = {:.3f}'.format(best_f1_val),
          'threshold = {:.3f}'.format(best_f1_threshold))
    return pd.concat([df,
                      pd.DataFrame([{'model_name': model_name,
                                     'lr': lr,
                                     'weight_decay': weight_decay,
                                     'hidden': hidden,
                                     'nb_heads': nb_heads,
                                     'dropout': dropout,
                                     'alpha': alpha,
                                     'loss': loss_val.data.item(),
                                     'accuracy': acc_val,
                                     'precision': prec_val,
                                     'recall': rec_val,
                                     'f1-score': f1_val,
                                     'best f1': best_f1_val,
                                     'threshold': best_f1_threshold}])],
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

# Perform evaluation for all weight files
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
    df = eval()

# Save evaluation results
df.to_csv('eval.csv', index=False)
