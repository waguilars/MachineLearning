import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

def get_recall(data):
    dimension = list(data.shape)
    recall=[]
    for i in range(dimension[0]):
            tp_fn_r=sum(data[i,:])
            tp_r=data[i][i]
            recall.append(tp_r/tp_fn_r)
    return recall

########### Test ##########################
y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
test = confusion_matrix(y_actu, y_pred)

tp_tn = sum(np.diag(test))
total = np.sum(test)

accuracy = tp_tn / total
print('manual accuracy: ', accuracy)
print('sklearn accuracy: ', accuracy_score(y_actu, y_pred))
print('sklearn presicion: ', precision_score(y_actu, y_pred, average=None))
print('sklearn recall: ', recall_score(y_actu, y_pred, average=None))
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

tp_tn = sum(np.diag(data))
total = np.sum(data)

# ==========================
#           RECALL
# ==========================
recall=get_recall(test)
print("recall: ",recall)
# ==========================
#         ACCURACY
# ==========================
accuracy = tp_tn / total
print("accuracy: ",accuracy)

