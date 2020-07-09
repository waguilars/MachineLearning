from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.utils import shuffle

# print(len(iris))
X, y = load_iris(return_X_y=True)

# X, y = shuffle(iris.data, iris.target, random_state=0)

#  50 primeras setosa  y 51 al 100  versicolor 
X = X[0:100]
y = y[0:100]

# division del dataset training 70% y test 30 % 
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)



print("#####################  Holdout method #####################")
model = LogisticRegression()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: %.2f" % (result*100.0))


print("####################  K-Fold Cross-Validation ##########################")
kfold = model_selection.KFold(n_splits=2,shuffle=True)
model_kfold = LogisticRegression()
model_kfold.fit(X_train, y_train)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f" % (results_kfold.mean()*100.0)) 

# Random Cross-Validation
print("####################  Random Cross-Validation ##########################")
RandomCV=model_selection.ShuffleSplit(n_splits=2,random_state=0)
model_random=LogisticRegression()
model_random.fit(X_train, y_train)
result_RandomCV=model_selection.cross_val_score(model_random,X_test,y_test,cv=RandomCV)
print("Accuracy: %.2f" % (result_RandomCV.mean()*100.0)) 

print("###################   Leave-one-out Cross-Validation ########################### ")
loocv = model_selection.LeaveOneOut()
model_loocv = LogisticRegression()
model_loocv.fit(X_train, y_train)
results_loocv = model_selection.cross_val_score(model_loocv, X_test, y_test, cv=loocv)
print("Accuracy: %.2f" % (results_loocv.mean()*100.0))