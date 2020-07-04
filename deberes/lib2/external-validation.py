"""
Integrantes:
    Wilson Aguilar
    Gabriel Cacuango
    Ricardo Romo
    Christian Lasso
"""
# !pip install validclust

import math as ma
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from validclust.indices import dunn, silhouette_score
from validclust.validclust import ValidClust
from sklearn.metrics import (pairwise_distances, adjusted_mutual_info_score,
                             adjusted_rand_score, normalized_mutual_info_score)


mtx = pd.read_pickle('mtx_icmla')

data = pd.read_csv('ICMLA.csv')

data.session = pd.Categorical(data.session)
labels = data.session.cat.codes.to_numpy()


dhc = AgglomerativeClustering(n_clusters=91, linkage='complete')
labelsp = dhc.fit_predict(mtx)
dist = pairwise_distances(mtx)


dendrogram = dendrogram(linkage(mtx, method='complete'))
model = AgglomerativeClustering(
    n_clusters=91, affinity='euclidean', linkage='complete')

plt.axhline(y=3.5, c='k')
plt.show()


dunn_index = dunn(dist, labelsp)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(mtx, labelsp)
print('Indice de silueta: ', silhouette_index)

nmi = normalized_mutual_info_score(labels, labelsp)
ami = adjusted_mutual_info_score(labels, labelsp)
ari = adjusted_rand_score(labels, labelsp)

print('NMI: ', nmi)
print('AMI: ', ami)
print('ARI: ', ari)

