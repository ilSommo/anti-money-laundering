__version__ = '1.0.0'
__author__ = 'Martino Pulici'


import torch
import torch.nn as nn
import torch.nn.functional as F

from aml.layers import SpGraphAttentionLayer


class GAT(nn.Module):
    """Graph ATtentional network.

    Attributes
    ----------
    dropout : float
        Dropout probability.
    attentions : list
        List of sparse matrix attention layers.
    out_att : aml.layers.SpGraphAttentionLayer
        Output sparse matrix attention layer.

    Methods
    -------
    __init__(self, nfeat, nhid, dropout, alpha, nheads)
        Initializes the class.
    forward(self, x, adj):
        Runs the module.

    """

    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """
        Initializes the class.

        Parameters
        ----------
        nfeat : int
            Number of input features.
        nhid : int
            Number of hidden layers.
        dropout : float
            Dropout probability.
        alpha : float
            LeakyReLU alpha parameter.
        nheads : int
            Number of attention heads.

        """
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
        """Runs the module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        adj : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output tensor.

        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.elu(self.out_att(x, adj))
        x = torch.sigmoid(x)
        return x


class GAT_MLP(nn.Module):
    """Graph ATtentional network with Multi Layer Perceptron.

    Attributes
    ----------
    dropout : float
        Dropout probability.
    attentions : list
        List of sparse matrix attention layers.
    out_att : aml.layers.SpGraphAttentionLayer
        Output sparse matrix attention layer.
    input_fc : torch.nn.Linear
        Fully connected input layer.
    hidden_fc : torch.nn.Linear
        Fully connected hidden layer.
    output_fc : torch.nn.Linear
        Fully connected output layer.

    Methods
    -------
    __init__(self, nfeat, nhid, dropout, alpha, nheads)
        Initializes the class.
    forward(self, x, adj):
        Runs the module.

    """

    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """
        Initializes the class.

        Parameters
        ----------
        nfeat : int
            Number of input features.
        nhid : int
            Number of hidden layers.
        dropout : float
            Dropout probability.
        alpha : float
            LeakyReLU alpha parameter.
        nheads : int
            Number of attention heads.

        """
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
        """Runs the module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        adj : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output tensor.

        """
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


class MLP(nn.Module):
    """ Multi Layer Perceptron.

    Attributes
    ----------
    dropout : float
        Dropout probability.
    input_fc : torch.nn.Linear
        Fully connected input layer.
    hidden_fc : torch.nn.Linear
        Fully connected hidden layer.
    output_fc : torch.nn.Linear
        Fully connected output layer.

    Methods
    -------
    __init__(self, nfeat, dropout)
        Initializes the class.
    forward(self, x, _):
        Runs the module.

    """

    def __init__(self, nfeat, dropout):
        """
        Initializes the class.

        Parameters
        ----------
        nfeat : int
            Number of input features.
        dropout : float
            Dropout probability.

        """
        super(MLP, self).__init__()
        self.dropout = dropout
        self.input_fc = nn.Linear(nfeat, nfeat * 10)
        self.hidden_fc = nn.Linear(nfeat * 10, 10)
        self.output_fc = nn.Linear(10, 1)

    def forward(self, x, _):
        """Runs the module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.

        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        x = self.output_fc(x)
        x = torch.sigmoid(x)
        return x


class MLP_GAT_MLP(nn.Module):
    """Graph ATtentional network with split Multi Layer Perceptron.

    Attributes
    ----------
    dropout : float
        Dropout probability.
    attentions : list
        List of sparse matrix attention layers.
    out_att : aml.layers.SpGraphAttentionLayer
        Output sparse matrix attention layer.
    input_fc : torch.nn.Linear
        Fully connected input layer.
    hidden_fc : torch.nn.Linear
        Fully connected hidden layer.
    output_fc : torch.nn.Linear
        Fully connected output layer.

    Methods
    -------
    __init__(self, nfeat, nhid, dropout, alpha, nheads)
        Initializes the class.
    forward(self, x, adj):
        Runs the module.

    """

    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """
        Initializes the class.

        Parameters
        ----------
        nfeat : int
            Number of input features.
        nhid : int
            Number of hidden layers.
        dropout : float
            Dropout probability.
        alpha : float
            LeakyReLU alpha parameter.
        nheads : int
            Number of attention heads.

        """
        super(MLP_GAT_MLP, self).__init__()
        self.dropout = dropout
        self.attentions = [
            SpGraphAttentionLayer(
                nfeat * 10,
                nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(
            nhid * nheads,
            nfeat * 10,
            dropout=dropout,
            alpha=alpha,
            concat=False)
        self.input_fc = nn.Linear(nfeat, nfeat * 10)
        self.hidden_fc = nn.Linear(nfeat * 10, 10)
        self.output_fc = nn.Linear(10, 1)

    def forward(self, x, adj):
        """Runs the module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        adj : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output tensor.

        """
        batch_size = x.shape[0]
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.view(batch_size, -1)
        x = F.relu(self.input_fc(x))
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.elu(self.out_att(x, adj))
        x = x.view(batch_size, -1)
        x = F.relu(self.hidden_fc(x))
        x = self.output_fc(x)
        x = torch.sigmoid(x)
        return x


class SLP(nn.Module):
    """ Multi Layer Perceptron.

    Attributes
    ----------
    dropout : float
        Dropout probability.
    linear : torch.nn.Linear
        Fully connected layer.

    Methods
    -------
    __init__(self, nfeat, dropout)
        Initializes the class.
    forward(self, x, _):
        Runs the module.

    """

    def __init__(self, nfeat, dropout):
        """
        Initializes the class.

        Parameters
        ----------
        nfeat : int
            Number of input features.
        dropout : float
            Dropout probability.

        """
        super(SLP, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(nfeat, 1)

    def forward(self, x, _):
        """Runs the module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.

        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
