import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# In this code, the Kaggle golf dataset is loaded using pandas, and the data is preprocessed
# by converting all columns to categorical data. The dataset is split into a training set
# with 10 data points and a testing set with 4 data points. A Multinomial Naive Bayes
# classifier is then initialized, trained on the training set, and used to predict
# outcomes for the test set. Finally, the model's training and testing accuracies
# are then evaluated and displayed. A minor adjustment was made to handle the capitalization
# mismatch in the 'play' column, which caused a KeyError during data splitting.

#%%

# Importing the dataset
df = pd.read_csv('golf_df.csv')
print(df.head())
# Getting Info
print(df.info())
# We have to convert everything into a category
df = df.apply(lambda x : x.astype('category'))
df1 = df.apply(lambda x : x.cat.codes)
print('\n')
print('New Dataframe: ')
print(df1.head())

#%%

# Let's divide into train and test split, since we have 14 datapoints in total, we split 10 for train and 4 for test

train = df1[:10]
test = df1[-4:]

y_train = train.pop('Play')
x_train = train

y_test = test.pop('Play')
x_test = test

print('\n')
print('Training Data: ')
print(x_train)
print('\n')
print('Testing Data: ')
print(x_test)

# Now multinomial navy bars, first initialize your model

model = MultinomialNB()
model_obj = model.fit(x_train,y_train)

#%%

y_out = model.predict(x_test)

#%%

print('\n')
print('Training Accuracy: ', model.score(x_train,y_train))
print('Testing Accuracy: ', model.score(x_test,y_test))
