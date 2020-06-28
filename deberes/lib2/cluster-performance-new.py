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
from sklearn.metrics import calinski_harabasz_score, pairwise_distances


dataset = load_iris()
iris = np.array(dataset.data)


#############################################
#			            Kmeans
#############################################

# |> Internal indexes
print('-----------    KMeans    ------------')
print('Indices de validacion interna ')
kmeans = KMeans(n_clusters=3, random_state=0).fit(iris)
labels = kmeans.fit_predict(iris)
dist = pairwise_distances(iris)

dunn_index = dunn( dist, labels)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(iris, labels)
print('Indice de silueta: ', silhouette_index)


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
