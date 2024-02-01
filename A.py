import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

mu1 = [2, 2]
sigma1 = [[0.9, -0.0255], [-0.0255, 0.9]]

mu2 = [5, 5]
sigma2 = [[0.5, 0], [0, 0.3]]

mu3 = [-2, -2]
sigma3 = [[1, 0], [0, 0.9]]

mu4 = [-4, 8]
sigma4 = [[0.8, 0], [0, 0.6]]

num=3000
cluster1 = np.random.multivariate_normal(mu1, sigma1, num)
cluster2 = np.random.multivariate_normal(mu2, sigma2, num)
cluster3 = np.random.multivariate_normal(mu3, sigma3, num)
cluster4 = np.random.multivariate_normal(mu4, sigma4, num)

stack=np.vstack([cluster1, cluster2, cluster3, cluster4])
scale=StandardScaler()
stack_scaled= scale.fit_transform(stack)

plt.scatter(stack[:, 0], stack[:, 1], c='black', s=5, marker='o', alpha=0.5)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

wcss=[]
for i in range(1, 21):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(stack_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 21), wcss)
plt.title("Elbow")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans=KMeans(n_clusters=4, init='k-means++', random_state=0)
y_kmeans=kmeans.fit_predict(stack_scaled)

plt.scatter(stack_scaled[y_kmeans == 0, 0], stack_scaled[y_kmeans== 0, 1], s=5, c= 'red', label='Cluster1', marker='+')
plt.scatter(stack_scaled[y_kmeans == 1, 0], stack_scaled[y_kmeans == 1, 1], s=5, c= 'blue', label='Cluster2', marker='^')
plt.scatter(stack_scaled[y_kmeans == 2, 0], stack_scaled[y_kmeans == 2, 1], s=5, c= 'yellow', label='Cluster3', marker='*')
plt.scatter(stack_scaled[y_kmeans == 3, 0], stack_scaled[y_kmeans == 3, 1], s=5, c= 'black', label='Cluster4', marker='s')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=7, label='Centroids', c='magenta')
plt.title("Clustering")
plt.legend()
plt.show()
