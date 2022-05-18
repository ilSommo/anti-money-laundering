__version__ = '1.0.0-alpha.1'
__author__ = 'Martino Pulici'


import torch


def best_f1(y_score, labels):
    y = zip(y_score.squeeze().data.tolist(), labels.data.tolist())
    y = sorted(y, key=lambda tup: tup[0])
    f1_scores = []
    for elem in y:
        t = elem[0]
        f1_scores.append(f1_score(y, t))
    return max(f1_scores)


def evaluate(output, labels, use_tensors=True):
    tp = tn = fp = fn = 0
    for i in range(len(labels)):
        if use_tensors:
            out = (torch.round(output[i]))
        else:
            out = (output[i])
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
