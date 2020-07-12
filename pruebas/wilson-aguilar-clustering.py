# !pip install validclust


import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from validclust.indices import dunn, silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (pairwise_distances, adjusted_mutual_info_score,
                             adjusted_rand_score, normalized_mutual_info_score)


data = pd.read_csv('segmentation_data.csv')
data = data[['frequency', 'recency', 'monetary']].values

################################################################
print('################  Kmeans #####################')
kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
dhc_labels = kmeans.fit_predict(data)

plt.scatter(data[dhc_labels == 0, 0], data[dhc_labels == 0, 1],
            s=50, c='cyan', label='Cluster 1')
plt.scatter(data[dhc_labels == 1, 0], data[dhc_labels == 1, 1],
            s=50, c='blue', label='Cluster 2')
plt.scatter(data[dhc_labels == 2, 0], data[dhc_labels == 2, 1],
            s=50, c='green', label='Cluster 3')  # VIP
plt.scatter(data[dhc_labels == 3, 0], data[dhc_labels == 3, 1],
            s=50, c='yellow', label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=75, c='red', label='Centroids')
plt.title('k-means frequency vs recency')
plt.show()
# 3
plt.scatter(data[dhc_labels == 0, 0], data[dhc_labels == 0, 2],
            s=50, c='cyan', label='Cluster 1')
plt.scatter(data[dhc_labels == 1, 0], data[dhc_labels == 1, 2],
            s=50, c='blue', label='Cluster 2')
plt.scatter(data[dhc_labels == 2, 0], data[dhc_labels == 2, 2],
            s=50, c='green', label='Cluster 3')  # VIP
plt.scatter(data[dhc_labels == 3, 0], data[dhc_labels == 3, 2],
            s=50, c='yellow', label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=75, c='red', label='Centroids')
plt.title('k-means frequency vs monetary')
plt.show()


print('tags: 2 -> VIP,  1 ->VIP Potencial, 3 -> Baja Frecuencia, 0 -> Nuevos')

dist = pairwise_distances(data)

# Los indices de validacion externa no podriamos sacar debido a
# que no conocemos los dhc_labels originales
print('\nIndices de validacion interna:')
dunn_index = dunn(dist, dhc_labels)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(data, dhc_labels)
print('Indice de silueta: ', silhouette_index)


################################################################
print('###################   DHC ##########################')
dhc = AgglomerativeClustering(n_clusters=4, linkage='complete')
dhc_labels = dhc.fit_predict(data)

dendrogram = dendrogram(linkage(data, method='complete'))
model = AgglomerativeClustering(
    n_clusters=4, affinity='euclidean', linkage='complete')
model.fit(data)
dhc_labels = model.fit_predict(data)


#plt.axhline(y=3.5, c='k')
# plt.show()

print('\nIndices de validacion interna:')
dunn_index = dunn(dist, dhc_labels)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(data, dhc_labels)
print('Indice de silueta: ', silhouette_index)

print('Pregunta 3:')
print('El rendimiento es mucho mejor ya que los indices de validacion para el caso del segundo algoritmo(DHC) es mas alta.')
