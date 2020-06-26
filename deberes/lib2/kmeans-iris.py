"""
Integrantes:
- Wilson Aguilar
- Gabriel Cacuango
- Christian Lasso
- Ricardo Romo
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import numpy as np

iris = np.array(load_iris().data)

wi = [None] * 11

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(iris)

    wi[i] = kmeans.inertia_

plt.plot(wi, marker='o')
plt.show()  # the best group size is 2 or 3





