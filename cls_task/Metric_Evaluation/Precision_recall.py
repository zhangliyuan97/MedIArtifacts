import numpy as np
from sklearn.metrics import  average_precision_score,auc
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from inspect import signature


def prc(targets, probas, precison=0.9):
    """
    Compute the precision and recall values for Classification's output, and plot P-R curve

    Args:
        targets (np.array): Ground Truth of Evaluation Dataset 
        probas (_np.array): Model's predicted probabilities of Evaluation Dataset
        precison (float, optional): _description_. Defaults to 0.9.

    Returns:
        _type_: a list contains precision, recall and corresponding threshold
    """
    avgpre = average_precision_score(targets, probas)
    pre, rec, thres = precision_recall_curve(targets, probas)
    index = np.argwhere(pre > precison)[0][0] # decreasing recall values
    if index == len(thres):
        irange = len(pre)-len(pre)//3
        index = np.argmax(pre[:irange])
        pre_i = pre[:irange][index]
        rec_i = rec[:irange][index]
        thres_i = thres[:irange][index]
    else:
        pre_i, rec_i, thres_i = pre[index], rec[index], thres[index]
    
    step_kwargs = ({'step': 'post'}
        if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(rec, pre, color='b', alpha=0.2, where='post')
    plt.fill_between(rec, pre, alpha=0.2, color='b', **step_kwargs)
    plt.plot([0, 1], [pre_i, pre_i], 'k--',
        label='@Thres{0:.3f}: Precision={1:.3f}, Recall={2:.3f}'.format(
        thres_i, pre_i, rec_i))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.title('PRC: AP={1:.3f}'.format(avgpre))
    plt.savefig('/PRC_epo_.png' )
    return pre_i, rec_i, thres_i