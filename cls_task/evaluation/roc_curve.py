from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

def roc_plot(targets, probas, auc):
    """
    Plot ROC curve

    Args:
        targets (np.array): Ground Truth of Evaluation Dataset 
        probas (_np.array): Model's predicted probabilities of Evaluation Dataset
        auc (float): auc values
    """
    fpr, tpr, _ = roc_curve(targets, probas)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('ROC')
    plt.savefig('./ROC.png')