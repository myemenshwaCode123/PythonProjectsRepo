import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing SVC and dataset from sklearn library
"""
When importing SVC (Support Vector Classification) and instantiating, I can mention
the type of kernel, if i happen to have a polynomial kernel, I can also mention the degree
"""
from sklearn.svm import SVC
from sklearn import datasets

# For visualizing the data, let's go easy and just extract the petal Width and petal Length
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # Petal Length, Petal Width
y = iris["target"]
# checking the type of flower, by converting the default dataset of
# multi-class classification, into binary classification
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# Let's create a scatter plot to visualize our dataset
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="class 1")
plt.legend()
plt.show()

# After visualizing, we can see it's a linear kernel
# Also since infinity is not allowed, we can just use a very large positive number for C
# such as 1e6, meaning it should be a hard classifier (no regularization)
# Which means I want the result 100%, no illusions or simplifying, a line clearly separating two classes
# Initializing the SVM Classifier model, then we fit our dataset

svm_clf = SVC(kernel="linear", C=1e6)  # Max c --> Hard Classifier
# Since SVM is a supervised machine learning model we have to specify both our input X and output y
svm_clf.fit(X, y)

# Weight terms
print(svm_clf.coef_)

# Bias term
print(svm_clf.intercept_)

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0 * x0 + w1 * x1 + b = 0
    # We have every term except x1 so we can use this formula --> x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    # Highlighting Support Vectors
    plt.scatter(svs[:,0], svs[:, 1], s = 180, facecolors='#FFAAAA', label="Support Vectors")
    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="Hyperplane")
    plt.plot(x0, gutter_up, "--", linewidth=2) # Use "--" for dashed line style
    plt.plot(x0, gutter_down, "--", linewidth=2)
    plt.legend()


plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], "bs")
plt.plot(X[y == 0][:, 0], X[y == 0][:, 1], "yo")
plt.xlabel("Petal Length", fontsize=14)
plt.axis([0, 5.5, 0, 2])

plt.show()

# Let's explore where exactly our support vectors are
print('\n')
print("Support Vectors Location: ")
print('\n')
print(svm_clf.support_vectors_)

# Lastly, let's see why Scaling is so important when it comes to SVM
Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
ys = np.array([0, 0, 1, 1])
svm_clf = SVC(kernel="linear", C=100)
svm_clf.fit(Xs, ys)

plt.figure(figsize=(9,2.7))
plt.subplot(121)
plt.plot(Xs[ys == 1][:, 0], Xs[ys == 1][:, 1], "bo")
plt.plot(Xs[ys == 0][:, 0], Xs[ys == 0][:, 1], "ms")
plot_svc_decision_boundary(svm_clf, 0, 6)
plt.xlabel("$x_0$", fontsize=20)
plt.ylabel("$x_1$   ", fontsize=20, rotation=0)
plt.title("Unscaled", fontsize=16)
plt.axis([0, 6, 0, 90])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xs)
svm_clf.fit(X_scaled, ys)

plt.subplot(122)
plt.plot(X_scaled[ys == 1][:, 0], X_scaled[ys == 1][:, 1], "bo")
plt.plot(X_scaled[ys == 0][:, 0], X_scaled[ys == 0][:, 1], "ms")
plot_svc_decision_boundary(svm_clf, -2, 2)
plt.xlabel("$x'_0$", fontsize=20)
plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)
plt.title("Scaled", fontsize=16)
plt.axis([-2, 2, -2, 2])

plt.show()

# Looking at the difference between the Unscaled data and the Scaled, you can easily see the significance
# Considering how tight the distance is between the hyperplane and each of the support vectors from
# each different class, more specifically how difficult it will be to separate those support vectors data points
# Unlike the scaled data
