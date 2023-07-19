import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# IRIS Data Set

# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and
# Iris versicolor), so 150 total samples.
# Let's use the Python Imaging Library to open and display images of each of the different species of flowers

# The Iris Setosa
import urllib.request
from PIL import Image

url1 = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
image1 = urllib.request.urlopen(url1)
img1 = Image.open(image1)

img1.show()

# The Iris Virginica
import urllib.request
from PIL import Image

url2 = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
image2 = urllib.request.urlopen(url2)
img2 = Image.open(image2)

img2.show()

# The Iris Versicolor
import urllib.request
from PIL import Image

url3 = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
image3 = urllib.request.urlopen(url3)
img3 = Image.open(image3)

img3.show()

# Let's import and print the data
iris = sns.load_dataset('iris')
print(iris.head())
# Now we will see whether we can use the KNN algorithm to actually classify these datapoints into these
# categories of flowers

# Explanatory Data Analysis, let's run a pair plot
sns.pairplot(iris,hue='species',palette='Dark2')
plt.show()
# When looking at the plot you will see the setosa specie is the most separable from the others

# Creating a Kernel Density Estimation function on the setosa flower to check what kind of distribution it has
# KDE plot of sepal_length versus sepal_width for setosa species of flower
setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(data=setosa, x='sepal_width', y='sepal_length', cmap="plasma", shade=True, shade_lowest=False)
plt.show()

# Standardizing the variables for everything except species since it's a categorical variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris.drop('species', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(iris.drop('species', axis=1))

# Now we convert the new dataset into a dataframe and view it to see the now normally distributed values
iris_feat = pd.DataFrame(scaled_features, columns=iris.columns[:-1])
print(iris_feat.head())

# Train Test Split, with training data being 70% and test data set to 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, iris['species'], test_size=0.30,random_state=103)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# Let's also evaluate the predictions
pred = knn.predict(X_test)

# Confusion Matrix and Classification Report
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
# Pretty high accuracy at 96% we will see if this is the best value for K

# Choosing the best K value
error_rate = []
for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Now plot the errors
plt.figure(figsize=(10,6))
plt.plot(range(1, 40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
# The error seems the lowest near 3-5 or 11, so let's test one of those values

# Retraining with a better K Value
# Now With K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('With K=3')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# So the accuracy improved to 100% with K=3 so it's safe to we found the best K Value