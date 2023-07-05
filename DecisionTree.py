import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder  # For train test splitting
from sklearn.model_selection import train_test_split  # For decision tree object
from sklearn.tree import DecisionTreeClassifier  # For checking testing results
from sklearn.metrics import classification_report, confusion_matrix  # For visualizing tree
from sklearn.tree import plot_tree

# Reading the data
iris = sns.load_dataset('iris')
print(iris.head(5))

# Let's do basic EDA on the dataset, we have to play with it
print(iris.info(5))
print(iris.shape)

# Check if there are any null values in the dataset
print(iris.isnull().any())

# Let's plot pair plot to visualize the attributes all at once
print(sns.pairplot(data=iris, hue='species'))

# Now let's check to see if there is a correlation or not (correlation matrix)
#print(sns.heatmap(iris.corr()))

# Now, we will separate the target variable(y) and features(X) as follows
target = iris['species']
df1 = iris.copy()
df1 = df1.drop('species', axis=1)

# Defining the attributes
X = df1
print(target)

# Now since Python only understands numbers, we must convert our variables into numbers
# Or in other words, label encoding as target variable is categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target = le.fit_transform(target)
print(target)
y = target

# Now, let's start splitting the dataset into training and test, 80% is training and 20% is test - 80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now I will make my decision tree and fit that into our train and test dataset
# Defining the decision tree algorithmdtree=DecisionTreeClassifier()
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')

# Now we will check the predicitions of our decision tree, or in other words, how accurate is it
# Predicting the values of test data, the higher the precision, the better the results will be
from sklearn.metrics import classification_report
y_prediction = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_prediction))

# Let's see if we can print our decision tree
from sklearn import tree
fn =['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn =['setosa','versicolor','virginica']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
tree.plot_tree(dtree,feature_names=fn,class_names=cn,filled=True)
fig.savefig('imagename.png')