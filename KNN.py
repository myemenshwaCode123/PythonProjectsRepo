import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Retrieve the data and initialize it into a variable
from sklearn.datasets import load_breast_cancer

import KNN

cancer = load_breast_cancer()

# The data is presented in a dictionary form
print(cancer.keys())

# Let's look at the description
print(cancer['DESCR'])

# Feature names
print(cancer['feature_names'])

# Setting up a dataframe
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_feat.info())

# Let's look at the target variable
# telling us whether a particular data point belongs to malignant or benign (cancerous or non-cancerous) (0 or 1)
print(cancer['target'])

# Let's convert the target variable into a dataframe
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
# How the datapoints look like
print(df_feat.head())

# Standardizing the variables and initialize the variable into "scaler"
# Essentially this means bringing all the samples around the same range
# We have to bring everything to the same scale to be able to compare the samples
# Then we fit the standardization or normalization on the data set we have, calculating the variance and means
# to apply it on the dataset to transform it to the actual values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_feat)
scaled_features = scaler.transform(df_feat)
# Let's view the scaled values
df_feat_scaled = pd.DataFrame(scaled_features,columns=df_feat.columns)
print(df_feat_scaled.head())

# Now let's divide the data into train and test with 30% being test and 70% being train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,np.ravel(df_target),test_size=0.30,random_state=105)
# importing the K-Nearest Neighbors classifier (KNN), the actual algorithm
from sklearn.neighbors import KNeighborsClassifier
# Initializing and fitting it into our data (K is equal to 1 for now, will see how results vary)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Now that we checked our accuracy lets run trial and error to find the best k value that gives us the best results
error_rate = []
for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
         markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# The plot reads the value 21 for K as the best value that produces the least amount of error
# Cant see plot on pycharm but on jupyter it works

# Now with K = 21
knn = KNeighborsClassifier(n_neighbors=21)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=21')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))