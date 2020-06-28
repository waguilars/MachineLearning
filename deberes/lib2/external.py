import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.datasets import load_iris

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

dataset =  load_iris()
iris = np.array(dataset.data)


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])

    return np.max(values)


def dunn(k_list):

    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di


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

# |> Índices de Validación Interna
#     Silueta
silhouette = silhouette_score(iris, y_kmeans, sample_size=50)
print('Indice de la silueta Kmeans: ', silhouette)


#############################################
#                   DHC
#############################################

dendrogram = dendrogram(linkage(iris, method='complete'))
model = AgglomerativeClustering(
    n_clusters=3, affinity='euclidean', linkage='complete')
model.fit(iris)
labels = model.fit_predict(iris)
plt.axhline(y=3.5, c='k')
# plt.show()

# clusters

# KMENAS CLUSTERS
pred = pd.DataFrame(y_kmeans)
pred.columns = ['Species']
df = pd.DataFrame(dataset.data)
prediction = pd.concat([df, pred], axis=1)
clus0 = prediction.loc[prediction.Species == 0]
clus1 = prediction.loc[prediction.Species == 1]
clus2 = prediction.loc[prediction.Species == 2]
k_list_clusters = [clus0.values, clus1.values, clus2.values]

# DHC CLUSTERS
predhc = pd.DataFrame(labels)
predhc.columns = ['Species']
predictiondhc = pd.concat([df, predhc], axis=1)
clus0dhc = predictiondhc.loc[predictiondhc.Species == 0]
clus1dhc = predictiondhc.loc[predictiondhc.Species == 1]
clus2dhc = predictiondhc.loc[predictiondhc.Species == 2]
dhc_list_clusters = [clus0dhc.values, clus1dhc.values, clus2dhc.values]


# |> Índices de Validación Interna
#############################################
#                   DUNN
#############################################
dunn_kmeans = dunn(k_list_clusters)
dunn_dhc = dunn(dhc_list_clusters)
print("indice de dunn KMEANS: ", dunn_kmeans)
print("indice de dunn DHC: ", dunn_dhc)


#############################################
#                   SILUETA
#############################################
silhouette = silhouette_score(iris, labels, sample_size=50)
print('Indice de la silueta DHC: ', silhouette)


plt.show()
print(y_kmeans)
print(load_iris().target)
print('NMI')


print(normalized_mutual_info_score(load_iris().target, y_kmeans))
print('AMI')

print(adjusted_mutual_info_score(load_iris().target, y_kmeans))
print('ARI')

print(adjusted_rand_score(load_iris().target, y_kmeans))
