__version__ = '1.0.0-alpha.1'
__author__ = 'Martino Pulici'


import numpy as np


def best_f1(y_score, labels):
    """Calculates the best f1-score.

    Parameters
    ----------
    y_score : torch.FloatTensor
        Predicted scores.
    labels : torch.FloatTensor
        Labels.

    Returns
    -------
    max_f1 : float
        Maximum f1-score.
    max_index : int
        Index of maximum f1-score.

    """
    y = zip(y_score.squeeze().data.tolist(), labels.data.tolist())
    y = sorted(y, key=lambda tup: tup[0])
    f1_scores = []
    for elem in y:
        t = elem[0]
        f1_scores.append(f1_score(y, t))
    max_f1 = max(f1_scores)
    max_index = y[np.argmax(f1_scores)][0]
    return max_f1, max_index


def evaluate(output, labels, threshold=0.5):
    """Calculates evaluation metrics.

    Parameters
    ----------
    output : torch.FloatTensor
        Predicted scores.
    labels : torch.FloatTensor
        Labels.
    threshold : float, default 0.5
        Threshold for positives.       

    Returns
    -------
    accuracy : float
        Accuracy.
    precision : float
        Precision.
    recall : float
        Recall.
    f1_score : float
        f1-score.

    """
    tp = tn = fp = fn = 0
    for i in range(len(labels)):
        out = (output[i]>=threshold).int()
        lab = (labels[i])
        if out == 1 and lab == 1:
            tp += 1
        elif out == 0 and lab == 0:
            tn += 1
        elif out == 0 and lab == 1:
            fn += 1
        else:
            fp += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1_score = 2 * (precision * recall) / max((precision + recall), 1)
    return accuracy, precision, recall, f1_score


def f1_score(y, t):
    """Calculates f1-score.

    Parameters
    ----------
    y : list
        Zipped outputs and labels.
    t : float
        Threshold for positives.

    Returns
    -------
    f1_score : float
        f1-score.

    """
    tp = tn = fp = fn = 0
    for i in range(len(y)):
        out = 1 if y[i][0] >= t else 0
        lab = y[i][1]
        if out == 1 and lab == 1:
            tp += 1
        elif out == 0 and lab == 0:
            tn += 1
        elif out == 0 and lab == 1:
            fn += 1
        else:
            fp += 1
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1_score = 2 * (precision * recall) / max((precision + recall), 1)
    return f1_score
