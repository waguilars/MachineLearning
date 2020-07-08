from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import LeaveOneOut

y=np.array(load_iris)
iris = np.array(load_iris().data)
iris=iris[0:100]
print(len(iris))
X, y=load_iris(return_X_y=True)

#  50 primeras setosa  y 51 al 100  versicolor 
X=X[0:100]
y=y[0:100]
""" print(X)
print("--------------------")
print(y) """

# division del dataset training 70% y test 30 % 
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.7,random_state=0)
""" print(X_test)
print(y_test) """

# implementaciones K=fold
print("------ K-fold -------")
kf = KFold(n_splits=2,shuffle=True)
for X_train, X_test in kf.split(X):
    print("%s %s" % (X_train, X_test))


print("----- Leave-one-out ----- ")
loo = LeaveOneOut()
print(loo.get_n_splits(X))
for train_index, test_index in loo.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # aqui nos imprime las 100 iteraciones
    # lo comentamos por que ocupa mucho espacio en la pantalla
    # print(X_train, y_train, y_test)
