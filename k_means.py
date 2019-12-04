#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# We will use the iris dataset
iris_dataset = skdata.load_iris()
X = iris_dataset.data  # (150, 4)
y = iris_dataset.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_hat = kmeans.labels_

# Visualize by projecting to lower dimensions
Z = PCA(n_components=3).fit_transform(X)

data_split = (Z[np.where(y_hat == 0)[0], :],
              Z[np.where(y_hat == 1)[0], :], Z[np.where(y_hat == 2)[0], :])
colors = ('blue', 'red', 'green')
labels = ('Setosa', 'Versicolour', 'Virginica')
markers = ('o', '^', '+')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
for z, c, l, m in zip(data_split, colors, labels, markers):
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
    ax.legend(loc='upper right')

# Let’s compare it to groundtruth labels
data_split_kmeans = (Z[np.where(y_hat == 0)[0], :],
                     Z[np.where(y_hat == 1)[0], :], Z[np.where(y_hat == 2)[0], :])
data_split_gt = (Z[np.where(y == 0)[0], :],
                 Z[np.where(y == 1)[0], :], Z[np.where(y == 2)[0], :])

data_splits = [data_split_kmeans, data_split_gt]
plot_titles = ['Partition by k-Means', 'Groundtruth']
fig = plt.figure()
for i in range(len(data_splits)):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.set_title(plot_titles[i])
    for z, c, l, m in zip(data_splits[i], colors, labels, markers):
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
        ax.legend(loc='upper right')
