"""
Integrantes:
- Wilson Aguilar
- Gabriel Cacuango
- Christian Lasso
- Ricardo Romo
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

iris = np.array(load_iris().data)

#############################################
#                   Kmeans
#############################################

kmeans = KMeans(n_clusters=3, random_state=0).fit(iris)
y_kmeans = kmeans.fit_predict(iris)

plt.scatter(iris[y_kmeans == 0, 0], iris[y_kmeans == 0, 1],
            s=50, c='cyan', label='Cluster 1')
plt.scatter(iris[y_kmeans == 1, 0], iris[y_kmeans == 1, 1],
            s=50, c='blue', label='Cluster 2')
plt.scatter(iris[y_kmeans == 2, 0], iris[y_kmeans == 2, 1],
            s=50, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=75, c='red', label='Centroids')
plt.title('k-means with 3 clusters')

# plt.show()

#############################################
#                   DHC
#############################################

dendrogram = dendrogram(linkage(iris, method='complete'))
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
model.fit_predict(iris)
labels = model.labels_
plt.axhline(y=3.5, c='k')
# plt.show()

##### clusters
kmeans_clusters=y_kmeans
dhc_clusters=labels

print(kmeans_clusters)
print(dhc_clusters)

