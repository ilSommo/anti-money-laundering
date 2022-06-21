__version__ = '1.0.0'
__author__ = 'Martino Pulici'


from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch


def check(name, unique):
    """Checks for name uniqueness.

    Parameters
    ----------
    name : str
        Name to check.
    unique : numpy.ndarray
        Array of unique names.

    Returns
    -------
    x
        Name.

    """
    if(name not in unique):
        name = -1
    return name


def load_data(data_dir='./data/elliptic_bitcoin_dataset/', random_state=28):
    """Loads data.

    Parameters
    ----------
    data_dir : str, default './data/elliptic_bitcoin_dataset/'
        Directory of data.
    random_state : int, default 28
        Random state seed.

    Returns
    -------
    adj : torch.FloatTensor
        Adjacent matrix.
    features : torch.FloatTensor
        Features matrix.
    labels : torch.FloatTensor
        Labels matrix.
    idx_train : torch.LongTensor
        Training set indexes.
    idx_val : torch.LongTensor
        Validation set indexes.
    idx_test : torch.LongTensor
        Test set indexes.

    """
    edges = pd.read_csv(data_dir + 'elliptic_txs_edgelist.csv')
    features = pd.read_csv(data_dir + 'elliptic_txs_features.csv', header=None)
    classes = pd.read_csv(data_dir + 'elliptic_txs_classes.csv')
    tx_features = ['tx_feat_' + str(i) for i in range(2, 95)]
    agg_features = ['agg_feat_' + str(i) for i in range(1, 73)]
    features.columns = ['txId', 'time_step'] + tx_features + agg_features
    features = pd.merge(
        features,
        classes,
        left_on='txId',
        right_on='txId',
        how='left')
    features = features[features['class'] != 'unknown']
    ratio = sum(features['class'] == '1') * 2 * 0.25 / len(features)
    features, X_test, _, _ = train_test_split(
        features, features['class'], stratify=features['class'], random_state=random_state, test_size=ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        features, features['class'], stratify=features['class'], random_state=random_state)
    under_sampler = RandomUnderSampler(random_state=random_state)
    X_train, _ = under_sampler.fit_resample(X_train, y_train)
    X_val, _ = under_sampler.fit_resample(X_val, y_val)
    idx_train = range(len(X_train))
    idx_val = range(len(X_train), len(X_train) + len(X_val))
    idx_test = range(
        len(X_train) +
        len(X_val),
        len(X_train) +
        len(X_val) +
        len(X_test))
    features = pd.concat([X_train, X_val, X_test])
    unique = features['txId'].unique()
    edges['txId1'] = edges['txId1'].apply(lambda name: check(name, unique))
    edges['txId2'] = edges['txId2'].apply(lambda name: check(name, unique))
    edges = edges[edges['txId1'] != -1]
    edges = edges[edges['txId2'] != -1]
    class_values = sorted(features['class'].unique())
    features_idx = {
        name: idx for idx,
        name in enumerate(
            sorted(
                features['txId'].unique()))}
    class_idx = {name: id for id, name in enumerate(class_values)}
    features['txId'] = features['txId'].apply(lambda name: features_idx[name])
    edges['txId1'] = edges['txId1'].apply(lambda name: features_idx[name])
    edges['txId2'] = edges['txId2'].apply(lambda name: features_idx[name])
    features['class'] = features['class'].apply(lambda name: class_idx[name])
    labels = features['class']
    classes = sorted(list(set(labels)), reverse=True)
    classes_dict = {
        c: np.identity(
            len(classes))[
            i,
            :] for i,
        c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    idx_features_labels = features.values[:, :-1]
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)
    edges_unordered = edges.values
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
        idx_features_labels.shape[0], idx_features_labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.FloatTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Normalizes adjacent matrix.

    Parameters
    ----------
    mx : torch.FloatTensor
        Adjacent matrix.

    Returns
    -------
    mx : torch.FloatTensor
        Adjacent matrix.

    """
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx


def normalize_features(mx):
    """Normalizes features matrix.

    Parameters
    ----------
    mx : torch.FloatTensor
        Features matrix.

    Returns
    -------
    mx : torch.FloatTensor
        Features matrix.

    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
