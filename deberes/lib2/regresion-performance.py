"""
Integrantes:
- Wilson Aguilar
- Gabriel Cacuango
- Christian Lasso
- Ricardo Romo
"""

import math as ma
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

y_predic = regresion.predict(X_test)


mae = mean_absolute_error(y_test, y_predic)
print("=================================")
print("MAE:", mae)
print("=================================\n")


mse = mean_squared_error(y_test, y_predic)
print("=================================")
print("MSE:", mse)
print("=================================\n")


rmse = ma.sqrt(mse)
print("=================================")
print("RMSE:", rmse)
print("=================================\n")

r2 = regresion.score(X_test, y_test)
print("=================================")
print("R2:", r2)
print("=================================\n")

r2a = 1-(1-r2)*(len(y_predic)-1)/(len(y_predic)-len(regresion.coef_)-1)
print("=================================")
print("R2 Ajustado:", r2a)
print("=================================\n")
