from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

x = load_iris().data[:, 0:3]
y = load_iris().data[:, 3]
print(x)
print(y)
print(load_iris().feature_names)
