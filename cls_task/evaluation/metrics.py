import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def metric(output, target):
    pred = np.argmax(output, axis=1)
    batch_size = target.shape[0]
    acc = float((pred == target).astype(int).sum()) / batch_size
    sen_list, spe_list = get_multi_cls_sen_spe(pred, target)
    return {'acc': acc, 'sen': sen_list, 'spe': spe_list}


def get_multi_cls_sen_spe(output, target):
    num_classes = np.unique(target).shape[0]
    batch_size = target.shape[0]
    multi_cls_confusion = confusion_matrix(target, output)
    sen_list, spe_list = [], []
    for cls_idx in range(num_classes):
        num_pos = (target == cls_idx).sum()
        num_neg = batch_size - num_pos
        num_tp = multi_cls_confusion[cls_idx][cls_idx]
        num_tn = multi_cls_confusion.trace() - num_tp
        sen = (num_tp / num_pos) if num_pos > 0 else 0.0
        spe = (num_tn / num_neg) if num_neg > 0 else 0.0
        sen_list.append((sen, num_pos))
        spe_list.append((spe, num_neg))

    return sen_list, spe_list


def multi_cls_roc_auc_score(prob, label):
    num_classes = prob.shape[1]
    auc_roc_ovr = []
    for cls_a in range(num_classes):
        for cls_b in range(num_classes):
            if cls_a != cls_b:
                cur_prob = prob.copy()
                cur_label = label.copy()
                a_mask = label == cls_a
                b_mask = label == cls_b
                ab_mask = np.logical_or(a_mask, b_mask)
                a_true = a_mask[ab_mask]
                a_true_score = roc_auc_score(a_true, cur_prob[ab_mask, cls_a])
                auc_roc_ovr.append(a_true_score)

    return auc_roc_ovr
