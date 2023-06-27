import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

jobs = pd.read_csv("Job Fair Survey.csv")
#print(jobs.head())
print(jobs.columns)

plt.figure(figsize=(16,8))
plt.scatter(
    jobs['How old are you?'],
    jobs['Have you been to a job fair, if so, how many?'],
    c='black'
)
plt.xlabel = ("Age")
plt.ylabel = ("Amount of Job Fairs Attended")
plt.show()

X = jobs['How old are you?'].values.reshape(-1,1)
Y = jobs['Have you been to a job fair, if so, how many?'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X,Y)

print(reg.coef_[0][0])
print(reg.intercept_[0])

predictions = reg.predict(X)

plt.figure(figsize=(16,8))
plt.scatter(
    jobs['How old are you?'],
    jobs['Have you been to a job fair, if so, how many?'],
    c='black'
)

plt.plot(
    jobs['How old are you?'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Age")
plt.ylabel("Amount of Job Fairs Attended")
plt.show()