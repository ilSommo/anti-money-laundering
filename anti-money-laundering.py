__version__ = '1.0.0-alpha'
__author__ = 'Martino Pulici'


import argparse
import glob
import os
import random
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from aml.load_data import load_data
from aml.models import GAT, GAT_MLP, MLP, SLP
from aml.utils import evaluate


def test():
    model.eval()
    output = model(features, adj)
    loss_test = loss(output[idx_test], labels[idx_test].unsqueeze(1))
    acc_test, prec_test, rec_test, f1_test = evaluate(
        output[idx_test], labels[idx_test])
    print('Test set results:',
          'loss = {:.3f},'.format(loss_test.data.item()),
          'accuracy = {:.3f},'.format(acc_test),
          'precision = {:.3f},'.format(prec_test),
          'recall = {:.3f},'.format(rec_test),
          'f1-score = {:.3f}'.format(f1_test))


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = loss(output[idx_train], labels[idx_train].unsqueeze(1))
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)
    loss_val = loss(output[idx_val], labels[idx_val].unsqueeze(1))
    acc_val, prec_val, rec_val, f1_val = evaluate(
        output[idx_val], labels[idx_val])

    if epoch % 10 == 0:
        print('Epoch {:04d}:'.format(epoch),
              'loss_train = {:.3f},'.format(loss_train.data.item()),
              'loss_val = {:.3f},'.format(loss_val.data.item()),
              'acc_val = {:.3f},'.format(acc_val),
              'prec_val = {:.3f},'.format(prec_val),
              'rec_val = {:.3f},'.format(rec_val),
              'f1_val = {:.3f},'.format(f1_val),
              'time = {:.3f} s'.format(time.time() - t))

    return loss_val.data.item()


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MLP', help='Model.')
parser.add_argument('--seed', type=int, default=28, help='Random seed.')
parser.add_argument(
    '--epochs',
    type=int,
    default=10000,
    help='Number of epochs to train.')
parser.add_argument(
    '--lr',
    type=float,
    default=0.005,
    help='Initial learning rate.')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='Weight decay (L2 loss on parameters).')
parser.add_argument(
    '--hidden',
    type=int,
    default=8,
    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.2,
    help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()

if args.model == 'SLP':
    model = SLP(nfeat=features.shape[1], dropout=args.dropout)

if args.model == 'MLP':
    model = MLP(nfeat=features.shape[1], dropout=args.dropout)

if args.model == 'GAT':
    model = GAT(
        nfeat=features.shape[1],
        nhid=args.hidden,
        dropout=args.dropout,
        alpha=args.alpha,
        nheads=args.nb_heads)

if args.model == 'GAT_MLP':
    model = GAT_MLP(
        nfeat=features.shape[1],
        nhid=args.hidden,
        dropout=args.dropout,
        alpha=args.alpha,
        nheads=args.nb_heads)

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
    lr=args.lr,
    weight_decay=args.weight_decay)
loss = nn.BCELoss()

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

for epoch in range(1, args.epochs + 1):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))

    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')

    for file in files:
        epoch_nb = int(file.split('.')[0])

        if epoch_nb < best_epoch:
            os.remove(file)

print('Optimization Finished!')
print('Total time elapsed: {:.3f}s'.format(time.time() - t_total))

print('Loading epoch {}'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

test()

torch.save(model.state_dict(),
           'checkpoints/{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.model,
                                                         args.lr,
                                                         args.weight_decay,
                                                         args.hidden,
                                                         args.nb_heads,
                                                         args.dropout,
                                                         args.alpha))

files = glob.glob('*.pkl')

for file in files:
    os.remove(file)
