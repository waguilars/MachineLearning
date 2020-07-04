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

mtx = pd.read_pickle('mtx_icmla')

data = pd.read_csv('ICMLA.csv')

data.session = pd.Categorical(data.session)
labels = data.session.cat.codes.to_numpy()
print(labels)

