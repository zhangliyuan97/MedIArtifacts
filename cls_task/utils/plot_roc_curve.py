import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_roc_curve(labels,preds,savepath):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path 
    """
    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)  ###计算真正率和假正率
    
    roc_auc1 = metrics.auc(fpr1, tpr1)  ###计算auc的值，auc就是曲线包围的面积，越大越好
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc1)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(savepath) #保存文件
