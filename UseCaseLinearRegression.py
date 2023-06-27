import matplotlib.pyplot as pit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error\

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