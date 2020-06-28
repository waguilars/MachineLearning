"""
Integrantes:
- Wilson Aguilar
- Gabriel Cacuango
- Christian Lasso
- Ricardo Romo
"""
# !pip install validclust

from validclust.indices import dunn, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.metrics import (pairwise_distances, adjusted_mutual_info_score,
                             adjusted_rand_score, normalized_mutual_info_score)


dataset = load_iris()
iris = np.array(dataset.data)


#############################################
#			            Kmeans
#############################################

# |> Internal indexes
print('-----------    KMeans    ------------')
print('Indices de validacion interna ')
kmeans = KMeans(n_clusters=3, random_state=0).fit(iris)
kmeans_labels = kmeans.fit_predict(iris)
y_kmeans = kmeans.fit_predict(iris)
dist = pairwise_distances(iris)

plt.scatter(iris[y_kmeans == 0, 0], iris[y_kmeans == 0, 1],
            s=50, c='cyan', label='Cluster 1')
plt.scatter(iris[y_kmeans == 1, 0], iris[y_kmeans == 1, 1],
            s=50, c='blue', label='Cluster 2')
plt.scatter(iris[y_kmeans == 2, 0], iris[y_kmeans == 2, 1],
            s=50, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=75, c='red', label='Centroids')
plt.title('k-means with 3 clusters')
plt.show()


dunn_index = dunn(dist, kmeans_labels)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(iris, kmeans_labels)
print('Indice de silueta: ', silhouette_index)

print('\nIndices de validacion externa:')
nmi = normalized_mutual_info_score(load_iris().target, kmeans_labels)
ami = adjusted_mutual_info_score(load_iris().target, kmeans_labels)
ari = adjusted_rand_score(load_iris().target, kmeans_labels)
print('NMI: ', nmi)
print('AMI: ', ami)
print('ARI: ', ari)



#############################################
#			            DHC
#############################################
print('\n-----------    DHC    ------------')
print('Indices de validacion interna')
dhc = AgglomerativeClustering(n_clusters=3, linkage='complete')
dhc_labels = dhc.fit_predict(iris)

dendrogram = dendrogram(linkage(iris, method='complete'))
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
model.fit(iris)
labels = model.fit_predict(iris)
plt.axhline(y=3.5, c='k')
plt.show()

dunn_index = dunn(dist, dhc_labels)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(iris, dhc_labels)
print('Indice de silueta: ', silhouette_index)

print('\nIndices de validacion externa:')
nmi = normalized_mutual_info_score(load_iris().target, dhc_labels)
ami = adjusted_mutual_info_score(load_iris().target, dhc_labels)
ari = adjusted_rand_score(load_iris().target, dhc_labels)
print('NMI: ', nmi)
print('AMI: ', ami)
print('ARI: ', ari)