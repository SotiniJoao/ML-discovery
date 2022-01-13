from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Manipulating the database

data, classes = load_wine(as_frame=True , return_X_y=True)

print("Data.describe:")
print(data.describe())

# Normalizing the data
scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

print("Normalized Data:")
print(X)

# Finding Clusters in the data with KMeans
kmeans1 = KMeans(n_clusters=2)
kmeans2 = KMeans(n_clusters=3)

kmeans1.fit(X)
kmeans2.fit(X)

clusters1 = kmeans1.predict(X)
clusters2 = kmeans2.predict(X)

print('Clusters:')
print('Cluster1 (n_clusters=2):')
print(clusters1)
print('Cluster2 (n_clusters=3):')
print(clusters2)

print('Finding the ideal number of clusters using adjusted rand score, given the predicted data and the real data:')
print("n_clusters=2: {}", metrics.adjusted_rand_score(classes,clusters1))
print("n_clusters=3: {}", metrics.adjusted_rand_score(classes,clusters2))

# building a cluster visualization model:

pca = PCA(n_components=2)
decomposed_X1 = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])
decomposed_X2 = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])
decomposed_X1['clusters'] = clusters1
decomposed_X2['clusters'] = clusters2
print("Decomposed Clusters using PCA:")
print("Decomposed n_clusters=2:")
print(decomposed_X1)
print("Decomposed n_clusters=3:")
print(decomposed_X2)

# Plotting the Cluster Charts
plt.figure(0)
plt.title("Clustering with n_clusters = 2")
plt.scatter(decomposed_X1[decomposed_X1['clusters']== 0].loc[:, 'PC1'], decomposed_X1[decomposed_X1['clusters']== 0].loc[:, 'PC2'], color='slateblue')
plt.scatter(decomposed_X1[decomposed_X1['clusters']== 1].loc[:, 'PC1'], decomposed_X1[decomposed_X1['clusters']== 1].loc[:, 'PC2'], color='springgreen')

plt.figure(1)
plt.title("Clustering with n_clusters = 3")
plt.scatter(decomposed_X2[decomposed_X2['clusters']== 0].loc[:, 'PC1'], decomposed_X2[decomposed_X2['clusters']== 0].loc[:, 'PC2'], color='slateblue')
plt.scatter(decomposed_X2[decomposed_X2['clusters']== 1].loc[:, 'PC1'], decomposed_X2[decomposed_X2['clusters']== 1].loc[:, 'PC2'], color='springgreen')
plt.scatter(decomposed_X2[decomposed_X2['clusters']== 2].loc[:, 'PC1'], decomposed_X2[decomposed_X2['clusters']== 2].loc[:, 'PC2'], color='gold')

# Agglomerative Clustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


hc = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

# Plotting the Dendrogram
X_hc = hc.fit(X)
plt.figure(2)
plot_dendrogram(X_hc, p=4, truncate_mode='level')
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")


# DBScan
# finding the nearest neighbors

print('The ideal DBScan EPS is given by the proximity between data points\n'
      'We can use a chart that gives us the ideal proximity')
print('The ideal proximity is given by the value in which the chart has its maximum curvature')
print('\n')
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(3)
plt.title("Finding the ideal DBScan EPS")
plt.plot(distances)
plt.grid()
# Starting the DBScan process
db = DBSCAN(eps=3, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Analysis of the DBScan Results:")
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(classes, labels))
print("Completeness: %0.3f" % metrics.completeness_score(classes, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(classes, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(classes, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(classes, labels)
)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

plt.show()
