import numpy as np
import seaborn as sns

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score


def get_accuracy(outputs, targets, cutoff_value=0.5):
    """Calculate the accuracy score for logistic regression problem.

    :param outputs: continous probability of [0, 1], np.nd_array
    :param targets: indicator of labels, 0/1, np.nd_array

    return accuracy
    """
    outputs_cutoff = outputs.copy()
    outputs_cutoff[outputs >= cutoff_value] = 1
    outputs_cutoff[outputs < cutoff_value] = 0

    return accuracy_score(targets, outputs_cutoff)


def get_specifity(outputs, targets, cutoff_value=0.5):
    """Calculate the specifity score for logistic regression problem.

    :param outputs: continous probability of [0, 1], np.nd_array
    :param targets: indicator of labels, 0/1, np.nd_array

    return speficity = TN / (TN + FP)
    """
    outputs_cutoff = outputs.copy()
    outputs_cutoff[outputs >= cutoff_value] = 1
    outputs_cutoff[outputs < cutoff_value] = 0

    # calculate TN, FP
    TN = ((outputs_cutoff == 0) & (targets == 0)).sum()
    FP = ((outputs_cutoff == 1) & ((1 - targets) == 1)).sum()

    specifity = TN / (TN + FP)

    return specifity


def get_sensitivity(outputs, targets, cutoff_value=0.5):
    """Calculate the sensitivity score for logistic regression problem.

    :param outputs: continous probability of [0, 1], np.nd_array
    :param targets: indicator of labels, 0/1, np.nd_array

    return sensitivity (recall) = TP / (TP + FN)
    """
    outputs_cutoff = outputs.copy()
    outputs_cutoff[outputs >= cutoff_value] = 1
    outputs_cutoff[outputs < cutoff_value] = 0

    return recall_score(targets, outputs_cutoff)

def get_auc(outputs, targets):
    return roc_auc_score(targets, outputs)