__version__ = '1.0.0-alpha.1'
__author__ = 'Martino Pulici'


from re import X
from turtle import xcor
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialSpmm(nn.Module):
    """Applies the SpecialSpmmFunction to a tensor.

    Methods
    -------
    forward(self, indices, values, shape, b)
        Runs the module.

    """

    def forward(self, indices, values, shape, b):
        """Runs the module.

        Parameters
        ----------
        indices : torch.Tensor
            First tensor indices.
        values : torch.Tensor
            First tensor values.
        shape : torch.Size
            Shape of the first tensor.
        b : torch.Tensor
            Second tensor to multiply.

        Returns
        -------
        x : torch.Tensor
            Result of the multiplication.

        """
        x = SpecialSpmmFunction.apply(indices, values, shape, b)
        return x


class SpecialSpmmFunction(torch.autograd.Function):
    """Calculates a matrix multiplication with a sparse tensor.

    Methods
    -------
    forward(ctx, indices, values, shape, b)
        Runs the module.
    backward(ctx, grad_output)
        Propagates backward.

    """

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        """Runs the module.

        Parameters
        ----------
        ctx : torch.autograd.function.SpecialSpmmFunctionBackward
            Backward function.
        indices : torch.Tensor
            First tensor indices.
        values : torch.Tensor
            First tensor values.
        shape : torch.Size
            Shape of the first tensor.
        b : torch.Tensor
            Second tensor to multiply.

        Returns
        -------
        x : torch.Tensor
            Result of the multiplication.
        
        """
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        x = torch.matmul(a, b)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Propagates backward.

        Parameters
        ----------
        ctx : torch.autograd.function.SpecialSpmmFunctionBackward
            Backward function.
        grad_output : torch.Tensor
            Gradient output.

        Returns
        -------
        grad_values : torch.Tensor
            Backward propagation of the tensor values.
        grad_b : torch.Tensor
            Backward propagation of the second tensor.
        
        """
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpGraphAttentionLayer(nn.Module):
    """Sparse matrix attention layer.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    alpha : float
        LeakyReLU alpha parameter.
    concat : bool
        Concatenation flag.
    W : torch.nn.parameter.Parameter
        Parameter matrix.
    a : torch.nn.parameter.Parameter
        Parameter matrix.
    leakyrelu : torch.nn.modules.activation.LeakyReLU
        LeaktReLU module.
    special_spmm : aml.layers.SpecialSpmm
        SpecialSpmm module.

    Methods
    -------
    __init__(self, in_features, out_features, dropout, alpha, concat=True)
        Initializes the class.
    forward(self, x, adj)
        Runs the module.
    __repr__(self)
        Represents the class.

    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        Initializes the class.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        dropout : float
            Dropout probability.
        alpha : float
            LeakyReLU alpha parameter.
        concat : bool, default True
            Concatenation flag.

        """
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

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
        dv = 'cuda' if x.is_cuda else 'cpu'
        N = x.size()[0]
        edge = adj.nonzero().t()
        h = torch.mm(x, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-torch.clamp(self.leakyrelu(
            self.a.mm(edge_h).squeeze()), min=-50, max=50))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size(
            [N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            x = F.elu(h_prime)
        else:
            x = h_prime  
        return x

    def __repr__(self):
        """Represents the class.

        Returns
        -------
        x : str
            Class representation.
        
        """
        x = self.__class__.__name__ + \
            ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
        return x
