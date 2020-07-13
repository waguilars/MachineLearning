from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

x = load_iris().data[:, 0:3]
y = load_iris().data[:, 3]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

regresion = LinearRegression().fit(X_train, y_train)
r2 = regresion.score(X_test, y_test)
