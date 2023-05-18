from matplotlib import pyplot as plt
from numpy import interp
import numpy as np
import seaborn as sns
from sklearn.metrics import *
from tensorflow.python.ops.confusion_matrix import confusion_matrix


def log(DEBUG, l):
    if DEBUG == 1: print(l)


def draw_CM(label, predicted):
    cm = confusion_matrix(label, predicted)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # true : false rate
    true = 0
    false = 0
    for i, j in enumerate(label):
        if j != predicted[i]:
            false += 1
        else: true += 1
    classification_report = classification_report(label, predicted)
    multilabel_to_binary_matrics = multilabel_confusion_matrix(label, predicted)

    return plt.show(), print('true rate: ', true), print('false rate: ', false), print(), \
           print('='*10, 'classification_report: ', '\n', classification_report), \
           print('='*10, 'multilabel_to_binary_matrics by class_num: ', '\n', '[[TN / FP] [FN / TP]]', '\n', multilabel_to_binary_matrics)


def draw_ROC_AUC(x, y, category_names):
    n_classes = len(category_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], x[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=1)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=1)

    colors = (['purple', 'pink', 'red', 'green', 'yellow', 'cyan', 'magenta', 'blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1, label=f'Class {i} ROC curve (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC & AUC')
    plt.legend(loc="lower right")

    return plt.show()
