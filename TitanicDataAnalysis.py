import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

# In this code I am exploring the titanic dataset more specifically the people on board, both those who survived
# and those who did not, ultimately trying to come to a better explanation and answer as to what factors
# made people more likely to survive the sinking of the Titanic, by looking at our titanic dataset and
# analyzing which factor would have been contributed the most to the chances of a person's survival
# on the ship or not, using logistic regression to predict survival.


#Gather your Data

titanic_data = pd.read_csv("Titanic.csv")
print(titanic_data.head())
print("# of passengers in original data: " + str(len(titanic_data.index)))

#Analyzing Data

#Around 550 pasengers did not survive and around 330 did survive,
#Concluding there are very less survivors than non-survivors
sns.countplot(x="Survived", data=titanic_data)
#A majority of males did not survive and a majority of females did survive
#It appears on average women were more than 3 times more likely to survive than men
sns.countplot(x="Survived", hue="Sex", data=titanic_data)
#Passengers who did not survive were majorly of third or lowest class
#Passengers traveling in 1st or 2nd class tend to survive more
sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
#Most of the passengers were in the young to average category for age
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)

#Data Wrangling

titanic_data.isnull()
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")
sns.boxplot(x="Pclass", y="Age", data=titanic_data)
print(titanic_data.head(5))
titanic_data.drop("Cabin", axis=1, inplace=True)
print(titanic_data.head(5))
titanic_data.dropna(inplace=True)
#Checking to see if your data is really clean
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False)
titanic_data.isnull().sum()
print(titanic_data.head(2))

# We have a lot of String values, so we need to convert them into categorical variables
# Let's use dummy variables, keeping in mind logistic regression only takes in 2 variables
sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
Pcl = pd.get_dummies(titanic_data['Pclass'], drop_first=True)

# Convert dummy variables to integers (0 and 1)
sex = sex.astype(int)
embark = embark.astype(int)
Pcl = Pcl.astype(int)

# Concatenate the dummy variables with the original dataframe
titanic_data = pd.concat([titanic_data, sex, embark, Pcl], axis=1)

# Drop unnecessary columns
titanic_data.drop(['Sex', 'Embarked', 'PassengerId', 'Pclass', 'Name', 'Ticket'], axis=1, inplace=True)

# Convert column names to strings
titanic_data.columns = titanic_data.columns.astype(str)

# Train Data
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Accuracy Check
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score: "), print(accuracy * 100)

