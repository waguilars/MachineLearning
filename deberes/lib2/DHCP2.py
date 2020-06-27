import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


data=pd.DataFrame(load_iris().data)

data_scaled = normalize(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# print(data_scaled.head())

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
shc.dendrogram(shc.linkage(data_scaled,method='single', metric='euclidean', optimal_ordering=False))
plt.axhline(y=0.40, color='r', linestyle='--')
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
cluster=cluster.fit_predict(data_scaled)
print(cluster)
