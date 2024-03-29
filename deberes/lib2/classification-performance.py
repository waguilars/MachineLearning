""" Integrantes:
  - Wilson Aguilar
  - Gabriel Cacuango
  - Ricardo Romo
  - Christian Lasso

"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score
)


def get_accuracy(conf_mtx):
    tp_tn = sum(np.diag(conf_mtx))
    total = np.sum(conf_mtx)
    return tp_tn / total


def get_recall(data):
    test = data.sum(axis=1)
    diagonal = np.diag(data)
    res = np.divide(diagonal, test)
    return res


def get_precision(data):
    test = data.sum(axis=0)
    diagonal = np.diag(data)
    res = np.divide(diagonal, test)
    return res


def get_fmeasure(conf_mtx):
    presicion = get_precision(conf_mtx)
    recall = get_recall(conf_mtx)
    num = presicion*recall
    den = presicion+recall
    f1 = 2*(num/den)
    return f1


########### Test ##########################
# y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
# y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
# test = confusion_matrix(y_actu, y_pred)
# print('manual accuracy: ', get_accuracy(test))
# print('sklearn accuracy: ', accuracy_score(y_actu, y_pred))
# print('sklearn presicion: ', precision_score(y_actu, y_pred, average=None))
# print('manual recall: ', get_recall(test))
# print('sklearn recall: ', recall_score(y_actu, y_pred, average=None))
# print('manual: f1', get_fmeasure(test))
# print('sklearn: f1', f1_score(y_actu, y_pred, average=None))


###################### deber ###################
print('-----------------   deber ----------')
data = np.array([
    [944, 0, 5, 3, 0, 12, 14, 1, 1, 0],
    [0, 1100, 4, 4, 1, 0, 2, 2, 22, 0],
    [20, 18, 873, 22, 18, 0, 20, 18, 40, 3],
    [10, 2, 29, 888, 2, 34, 2, 18, 19, 6],
    [1, 1, 5, 0, 893, 1, 20, 3, 6, 52],
    [21, 6, 3, 47, 11, 721, 26, 9, 41, 7],
    [20, 2, 12, 1, 18, 15, 883, 0, 7, 0],
    [5, 20, 27, 2, 8, 1, 0, 935, 4, 26],
    [7, 19, 11, 24, 11, 46, 29, 14, 792, 21],
    [10, 2, 2, 12, 53, 14, 1, 35, 14, 886]
])
# ==========================
#         ACCURACY
# ==========================
accuracy = get_accuracy(data)
print("accuracy: \n\t", accuracy)

# ==========================
#           PRESICION
# ==========================
presicion = get_precision(data)
print("presicion: \n\t", presicion)

# ==========================
#           RECALL
# ==========================
recall = get_recall(data)
print("recall: \n\t", recall)

# ==========================
#         F-MEASURE
# ==========================
f1 = get_fmeasure(data)
print("f-measure: \n\t", f1)
