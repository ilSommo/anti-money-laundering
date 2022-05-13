__version__ = '1.0.0-alpha'
__author__ = 'Martino Pulici'


import torch


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
