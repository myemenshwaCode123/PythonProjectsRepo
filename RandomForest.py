import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset('penguins')
print(df.head())
print(df.shape) # Let's check the amount of rows and columns in our dataset
print(df.info()) # Get some info about our dataset such as the different datatypes for each feature
print(df.isnull().sum()) # Let's calculate how many null values there are

# Drop the null values and check
df.dropna(inplace=True)
print(df.isnull().sum())

# Feature Engineering
# we have to use one hot encoding before feeding our data into the random forest algorithm
# because, we have categorical or object data types that need to be transformed into numeric data
# Below I applied in the sex column to output unique values
print(df.sex.unique())
# Use the Pandas library to get dummies
print(pd.get_dummies(df['sex'].head()))
# You can drop the first column for Female because its redundant and if it's not a male it's a female
sex = pd.get_dummies(df['sex'], drop_first=True)
print(sex.head())
# Repeat the same process for the feature Island
print(df.island.unique())
print(pd.get_dummies(df['island']).head())
# We can drop the first column since if its not dream or togersen it will be biscoe
island = pd.get_dummies(df['island'], drop_first=True)
print(island.head())
# Let's convert to 0 and 1 instead of false and true
sex = sex.astype(int)
island = island.astype(int)
print(sex.head())
print(island.head())

# Now let's inculde our two independent dataframes into our main dataframe
# In other words, concatenate the above two data frames to the original dataframe
new_data = pd.concat([df,island, sex], axis=1)
print(new_data.head())

# Drop the repeated columns
new_data.drop(['sex','island'], axis=1, inplace=True)
print(new_data.head())

# Create a separate target variable for species
Y = new_data.species
print(Y.head())
# Print the three unique values of the penguin species
print(Y.unique())
# The unique values are of the object data type, so we have to convert them into numerical data
# Let's use the map function to convert categorical values into numeric
Y = Y.map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
print(Y.head())
# Now let's drop the target variable species from our main dataframe
new_data.drop('species', inplace=True, axis=1)
print(new_data.head())
# Let's store our new data into X
X = new_data

# Splitting the Data set into Training and Test Data
from sklearn.model_selection import train_test_split
# We are splitting our training data into 70% and our Testing data into 30%
# The random state means you're not fixing any random state, also working for code reproducibility
# Meaning you will always get the same result or output, not changing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# Let's check our splitting by printing out the amount of values and features
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('Y_train', y_train.shape)
print('Y_test', y_test.shape)

# Training Random Forest Classification on Training Set
from sklearn.ensemble import RandomForestClassifier
# n_estimator just means we are creating some decision trees
classifier = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
print(classifier.fit(X_train, y_train))

# Predicting the Test Results
y_pred = classifier.predict(X_test)
print(y_pred)

# Now lets use a confusion matrix to check the accuracy of the random forest algorithm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Now let's check the accuracy score
print('Accuracy Score: ',accuracy_score(y_test, y_pred) * 100, '%')

# We have a very good accuracy score, with only two cases being misclassified by the random forest classifier

# let's print out the classification report
print(classification_report(y_test, y_pred))

# We have good recall and f1-score values from our report, but what if we changed the criteria to gini
# instead of entropy, as well as the amount of decision trees

# Experimenting
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=7, criterion='gini', random_state=0)
classifier.fit(X_train,y_train)
print(classifier.fit(X_train,y_train))
y_pred = classifier.predict(X_test)
print('Accuracy Score: ', accuracy_score(y_test, y_pred) * 100, '%')

# We were able to get a higher accuracy score by experimenting on our random forest classifier, more specifically
# how it works with several trees and different criteria, to produce a higher accuracy on our training and test data
