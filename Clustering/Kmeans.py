import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv(r"Mall_Customers.csv", encoding="UTF-8")
X=dataset.iloc[:, [3,4]].values


from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow")
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
y_kmeans = kmeans.fit_predict(X)
#print(y_kmeans)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans== 0, 1], s=50, c= 'red', label='Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c= 'blue', label='Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c= 'yellow', label='Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c= 'black', label='Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c= 'orange', label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='magenta', label='centroids')
plt.title("Clustering ")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()




