import matplotlib.pyplot as pit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error\

#This PyCharm project was my first project using python that dealt with data science.
#This Python project utilizes linear regression to predict diabetes progression based on a single feature.
#The code begins by importing necessary libraries for data visualization, numerical operations, and machine learning.
#The diabetes dataset is loaded using scikit-learns load diabetes () function.
#The data is prepared by extracting the desired feature and splitting it into training and testing sets.
#A linear regression model is then trained on the training data.
#Predictions are made on the test data, and the accuracy is evaluated using mean squared error.
#The learned coefficients and intercept of the regression model are printed.
#The results are visualized through scatter and line plots, displaying the test data and the predicted values.

disease = datasets.load_diabetes()

#print(disease)
disease_X = disease.data[:, np.newaxis,2]
#Splitting the data
disease_X_train = disease_X[:-30]
disease_X_test = disease_X[-20:]

disease_Y_train = disease.target[:-30]
disease_Y_test = disease.target[-20:]

reg = linear_model.LinearRegression()
reg.fit(disease_X_train,disease_Y_train)

y_predict = reg.predict(disease_X_test)

accuracy = mean_squared_error(disease_Y_test,y_predict)

print(accuracy)

weights = reg.coef_
intercept = reg.intercept_
print(weights,intercept)

plt.scatter(disease_X_test,disease_Y_test)
plt.plot(disease_X_test,y_predict)
plt.show()