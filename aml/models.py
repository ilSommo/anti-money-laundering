__version__ = '1.0.0-alpha'
__author__ = 'Martino Pulici'


import torch
import torch.nn as nn
import torch.nn.functional as F

from aml.layers import SpGraphAttentionLayer


class GAT_MLP(nn.Module):

    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT_MLP, self).__init__()
        self.dropout = dropout
        self.attentions = [
            SpGraphAttentionLayer(
                nfeat,
                nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(
            nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False)

        self.input_fc = nn.Linear(nfeat, nfeat * 10)
        self.hidden_fc = nn.Linear(nfeat * 10, 10)
        self.output_fc = nn.Linear(10, 1)

    def forward(self, x, adj):
        batch_size = x.shape[0]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.elu(self.out_att(x, adj))
        x = x.view(batch_size, -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        x = self.output_fc(x)
        x = torch.sigmoid(x)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [
            SpGraphAttentionLayer(
                nfeat,
                nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(
            nhid * nheads, 1, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.elu(self.out_att(x, adj))
        x = torch.sigmoid(x)
        return x


class MLP(nn.Module):

    def __init__(self, nfeat, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.input_fc = nn.Linear(nfeat, nfeat * 10)
        self.hidden_fc = nn.Linear(nfeat * 10, 10)
        self.output_fc = nn.Linear(10, 1)

    def forward(self, x, _):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        x = self.output_fc(x)
        x = torch.sigmoid(x)
        return x


class SLP(nn.Module):

    def __init__(self, nfeat, dropout):
        super(SLP, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(nfeat, 1)

    def forward(self, x, _):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
