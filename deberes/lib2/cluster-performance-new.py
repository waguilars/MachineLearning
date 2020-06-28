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
dist = pairwise_distances(iris)

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
