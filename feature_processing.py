import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn as sk
from sklearn import decomposition, preprocessing
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import functions created for this course
from def_functions import *

#path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"

# ---- Load ----
df = pd.read_csv(os.path.join(path_hc_baseline_eeg, "eeg_features.csv"))

del df['baseline']

df = df.iloc[1:]

X = df[['patient_id', 'n-back', 'delta', 'theta', 'alpha', 'se', 'pe', 'zc']]

X = X.fillna(X.mean()) 

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Create a k-means clustering model
kmeans = KMeans(n_clusters=2, max_iter=100,random_state = 0, n_init = "auto")
# Fit the data to the model
print(kmeans)
# Fit the data to the model
kmeans.fit(X_scaled)
# Determine which clusters each data point belongs to:
clusters =  kmeans.predict(X_scaled)
# Add cluster number to the original data
X_scaled_clustered = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
X_scaled_clustered['cluster'] = clusters

print(X_scaled_clustered.head())

# Run a number of tests, for 1, 2, ... num_clusters
num_clusters = 50
kmeans_tests = [KMeans(n_clusters=i, init='random', n_init=10) for i in range(1, num_clusters)]
score = [kmeans_tests[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans_tests))]

# Plot the curve
plt.plot(range(1, num_clusters),score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

from sklearn.decomposition import PCA

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Transfor the scaled data to the new PCA space
X_reduced = pca.transform(X_scaled)

# Convert to a data frame
X_reduceddf = pd.DataFrame(X_reduced, index=X.index, columns=['PC1','PC2'])
X_reduceddf['cluster'] = clusters

centres_reduced = pca.transform(kmeans.cluster_centers_)
display_factorial_planes(X_reduced, 2, pca, [(0,1)], illustrative_var = clusters, alpha = 0.8)
plt.scatter(centres_reduced[:, 0], centres_reduced[:, 1],
            marker='x', s=169, linewidths=3,
            color='k', zorder=10)
plt.show()

# Add the cluster number to the original scaled data
X_clustered = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
X_clustered["cluster"] = clusters

# Display parallel coordinates plots, one for each cluster
display_parallel_coordinates(X_clustered, 3)
plt.show()

# Create a data frame containing our centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
centroids['cluster'] = centroids.index

display_parallel_coordinates_centroids(centroids, 10)
plt.show()

