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


mtx = pd.read_pickle('mtx')

dhc = AgglomerativeClustering(n_clusters=5, linkage='complete')
labels = dhc.fit_predict(mtx)
print(labels)



