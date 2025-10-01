import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state= 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters= 5, init='k-means++', random_state= 42)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1], s=100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1], s=100, c = 'cyan', label = 'Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1], s=100, c = 'magenta', label = 'Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1], s=100, c = 'green', label = 'Cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1], s=100, c = 'blue', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c="yellow", label = 'Centroids')
plt.title('Cluster of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-1000)')
plt.legend()
plt.show()

# # 📌 K-Means and K-Means++ Notes

# ## 🔹 What is K-Means?
# - K-Means is an **unsupervised ML clustering algorithm**.
# - Goal: Partition dataset into **K clusters** such that:
#   - Each point belongs to the nearest cluster center (centroid).
#   - Minimize the **within-cluster sum of squares (WCSS)** → also called **inertia**.

# ---

# ## 🔹 Steps in K-Means Algorithm
# 1. Choose **K** (number of clusters).
# 2. **Initialize centroids** randomly (pick K random points).
# 3. **Assign step**: Assign each point to nearest centroid (using Euclidean distance).
# 4. **Update step**: Recalculate centroids as mean of cluster points.
# 5. Repeat **Assign → Update** until:
#    - Centroids stop changing, OR
#    - Max iterations reached.

# ---

# ## 🔹 Objective Function
# K-Means minimizes the **WCSS (Within-Cluster Sum of Squares)**:

# \[
# J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
# \]

# Where:  
# - \(C_i\) = cluster i  
# - \(\mu_i\) = centroid of cluster i  

# ---

# ## 🔹 Limitations of K-Means
# - Sensitive to **initial centroids** (different runs → different results).
# - Need to pre-define **K**.
# - Struggles with:
#   - Non-spherical clusters.
#   - Different cluster sizes/densities.
#   - Outliers.

# ---

# ## 🔹 K-Means++ Initialization
# Improves centroid initialization.

# ### Algorithm:
# 1. Pick **first centroid randomly** from dataset.
# 2. For each point `x`, compute **D(x)** = distance from nearest chosen centroid.
# 3. Pick next centroid with probability proportional to `D(x)^2`.
#    - (Farther points have higher chance to become centroids).
# 4. Repeat until K centroids chosen.
# 5. Run standard K-Means.

# ---

# ## 🔹 Advantages of K-Means++
# - Better starting centroids.
# - Faster convergence.
# - More consistent + better clusters.

# ---

# ## 🔹 Choosing K
# - **Elbow Method**: Plot WCSS vs K → find “elbow point”.
# - **Silhouette Score**: Measures cluster quality.

# ---

# ## ✅ Summary
# - **K-Means**: Simple, fast, works well for spherical/equal clusters.  
# - **K-Means++**: Smarter initialization → better results.
