from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Set visualization style
sns.set()
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('movie_metadata.csv')
print(data.shape)
print(data.head())

# Let's view the amount of director Facebook likes for each movie
print(data['director_facebook_likes'])

# Filtering out records we want to view using index location and assigning them to the "newdata" object
print("\n")
print(data.columns)
# Slicing the columns from 4 to 6 (including 4, excluding 6)
newdata = data.iloc[:, 4:6]
print(newdata)

# Handle missing values using the SimpleImputer
imputer = SimpleImputer(strategy='mean')
newdata_imputed = imputer.fit_transform(newdata)

# Using the KMeans from sklearn
# Specifying the number of clusters for KMeans
# Setting n_init to supress future warning that is being outputted
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(newdata_imputed)

# Print the number of cluster centers
print(len(kmeans.cluster_centers_))
# Print the labels assigned to each data point
print(kmeans.labels_)
# Print the number of labels (should match the number of data points)
print(len(kmeans.labels_))
# Print the type of labels (numpy array)
print(type(kmeans.labels_))
# Count the number of occurrences of each label
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts)))

# Add the cluster labels to the data
newdata['cluster'] = kmeans.labels_

# Plot the data using lmplot
sns.set_style('whitegrid')
sns.lmplot(x='director_facebook_likes', y='actor_3_facebook_likes', data=newdata, hue='cluster',
           palette='viridis', height=6, aspect=1, fit_reg=False, scatter_kws={'s': 50, 'alpha': 0.7})

# Show the plot
plt.show()

# Using the plot you can easily identify trends and patterns within each cluster, such as in, cluster 0, you can see
# a lot of movies are being made with new directors that don't get that many likes, since there not well know,
# with new actors, that are also not well known. In cluster 1, average directors are making films with new actors.
# Another trend would be in cluster 2, a lot of famous actors who are well known, getting a lot of likes,
# are doing movies with new directors, that don't have a lot of likes. In cluster 3 there are a couple famous
# directors making movies with famous actors, and lastly, in cluster 4, there are way more occurrences
# of famous directors making movies with new actors

