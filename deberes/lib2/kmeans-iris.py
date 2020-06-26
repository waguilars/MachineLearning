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

import numpy as np

iris = np.array(load_iris().data)

wi = [None] * 11

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(iris)

    wi[i] = kmeans.inertia_

plt.plot(wi, marker='o')
plt.title('Elbow - Numero de clusters')
plt.show()  # the best group size is 2 or 3

## Plot with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(iris)
y_kmeans = kmeans.fit_predict(iris)

plt.scatter(iris[y_kmeans==0, 0], iris[y_kmeans==0, 1], s=50, c='cyan', label ='Cluster 1')
plt.scatter(iris[y_kmeans==1, 0], iris[y_kmeans==1, 1], s=50, c='blue', label ='Cluster 2')
plt.scatter(iris[y_kmeans==2, 0], iris[y_kmeans==2, 1], s=50, c='green', label ='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=75, c='red', label = 'Centroids')
plt.title('k-means with 3 clusters')

plt.show()





