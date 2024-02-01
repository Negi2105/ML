import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv(r"C:\\Users\\91769\\OneDrive - Plaksha University\\Documents\\GitHub\\ML\\Clustering\\Mall_Customers.csv", encoding="UTF-8")
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





