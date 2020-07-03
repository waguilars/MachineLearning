""" 
Integrantes:
    Wilson Aguilar
    Gabriel Cacuango
    Ricardo Romo
    Christian Lasso
"""
import math as ma
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from validclust.indices import dunn, silhouette_score

mtx = pd.read_pickle('mtx')

dhc = AgglomerativeClustering(n_clusters=5, linkage='complete')
labels = dhc.fit_predict(mtx)

dendrogram = dendrogram(linkage(mtx, method='complete'))
model = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='complete')

plt.axhline(y=3.5, c='k')
plt.show()


dunn_index = dunn(mtx.to_numpy(), labels)
print('Indice de dunn: ', dunn_index)

silhouette_index = silhouette_score(mtx, labels)
print('Indice de silueta: ', silhouette_index)
