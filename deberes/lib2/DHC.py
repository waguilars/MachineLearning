"""
Integrantes:
- Wilson Aguilar
- Gabriel Cacuango
- Christian Lasso
- Ricardo Romo
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = np.array(load_iris().data)
wi = [None] * 11

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(iris)

    wi[i] = kmeans.inertia_

kmeans = KMeans(n_clusters=3, random_state=0).fit(iris)
y_kmeans = kmeans.fit_predict(iris)
print("KMeans")
print(y_kmeans)




