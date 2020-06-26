# Importing the necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Defining the Seaborn figure style
sns.set(style='white', color_codes=True)

# Importing warnings to alert the user of some condition in the program
import warnings

warnings.filterwarnings("ignore")

# Reading the data from the file CC.csv
data = pd.read_csv('./CC.csv')

# Part 1:
# Check how many clusters => 7
print(data['TENURE'].value_counts())

# Look for nulls
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# MINIMUM_PAYMENTS and CREDIT_LIMIT have nulls, below replacing them by the mean
data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].mean()
data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].mean()
nulls = pd.DataFrame(data.isnull().sum())
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# Dividing data into x, y where y is TENURE
x = data.iloc[:, 1:]
print(x.shape)

# Elbow Method
# wcss:within-cluster sums of squares
wcss = []
# try 1-9 clusters
print()
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    # Save within-cluster sums of squares to the list
    wcss.append(kmeans.inertia_)

# Display the graph
print(wcss)
plt.plot(range(1, 10), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Part 2
# From the map, at k=3 it seems like data slowly remains unchange so choose k=3
# Silhouette score
km = KMeans(n_clusters=3)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print()
print('Silhouette score for', 3, 'clusters', score)

# Part 3
# feature scaling to improve the Silhouette Score
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)

km = KMeans(n_clusters=3)
km.fit(X_scaled)
y_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('Silhouette score for', 3, 'clusters after scaling', score)

# Part 4
# Applying PCA on the same data set
#scale the features
scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)
pca= PCA(2)
x_pca= pca.fit_transform(x_scaler)
print("#"*50)

# Bonus part-1 PCA + KMeans
km1 =KMeans(n_clusters=3, random_state=0)
km1.fit(x_pca)
y_cluster_kmeans1= km1.predict(x_pca)
pca_score = metrics.silhouette_score(x_pca, y_cluster_kmeans1)
print("PCA + Kmeans Score is :", pca_score)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y_cluster_kmeans1)
plt.title('PCA + Kmeans')
plt.show()

# Bonus part-1 with PCA + Kmeans + scaling (For scaling i uses x_scalar instead of direct data(x))
x_pcascale = pca.fit_transform(x_scaler)
km = KMeans(n_clusters=3)
km.fit(x_pcascale)
Y_cluster_kmeans= km.predict(x_pcascale)
pca_means_scale_score = metrics.silhouette_score(x_pcascale, Y_cluster_kmeans)
print('PCA+KMEANS+ Scale score is:', pca_means_scale_score)

# Bonus part-2 visualizing the clustering with graph
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = Y_cluster_kmeans)
plt.title('PCA + Kmeans + Scaling')
plt.show()
