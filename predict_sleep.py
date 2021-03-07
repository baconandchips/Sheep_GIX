# import os
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# import datetime 
# import time
import joblib # check if we need it later


'''
Read file ✅
Grab heart rate, epoch ✅
Normalize Epoch linearly ✅
Normalize heart rate with gaussian distribution; probably Z-score ✅
Label sleep stages W N1 N2 N3 R as 0 1 2 3 4 (Label Encoding) ✅
Train and predict label on sleep stages
- What types of algorithms would be most useful in this? KNN OR SVM?
    - SVM usually works in different cases; how much computing resources do you have? KNN is resource intensive and takes lots of resources; if not then SVM or Decision tree. Medical program, decision tree may be preferred, because you can plot out the tree and figure out why the machine plotted it out like that. And even with one method, you need to fine tune the parameters so that it can work best. It really depends.
    - Lecture 4: goes over which algorithm to choose.
- Where can we reference reading in multiple datasets again, for one model's training? (There are 10 peoples' data that we can use for training, this is only one person's)
- Cross-use validation, lecture 5; take a look there! If ML works on every user, 8 users for training, 2 for test.
'''

df = pd.read_excel('./1_2.xlsx')

print("\n\nRaw dataframe:\n")
print(df)
print(df["HR"][0][0:2].strip()) # This gives first 3 chars without spaces- our heart rate without words attached to it.

def clean_heartrate(hr):
    # Clean away footnotes
    # print(type(hr))
    if type(hr) is str:
        return int(hr[0:2].strip())
    return hr

def normalize_column_mm(values):
    # Min Max normalization
    # print(type(values))
    # values_int = values.apply(clean_heartrate)
    min = np.min(values)
    max = np.max(values)
    norm = (values - min)/(max - min)
    return (pd.DataFrame(norm))

# Adapted from https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
def normalize_column_z(values):
    # Z-score normalization
    norm = (values - values.mean()) / values.std()
    return (pd.DataFrame(norm))

df["Epoch"] = normalize_column_mm(df["Epoch"])
df["HR"] = normalize_column_z(df["HR"].apply(clean_heartrate))
# df["HR"] = df["HR"].apply(clean_heartrate)

# Drop the "U" data rows, they're not something we need. Not a sleep state, and only 1 row out of all, for subject 1.
df = df[df["Stage"] != 'U']

print("\n\nNormalized dataframe:\n")
print(df)


# Now, let's make sleep stages numeric labels instead!
label_encoder = preprocessing.LabelEncoder()
print(df["Stage"].unique())
df["Stage"] = label_encoder.fit_transform(df["Stage"])
print(df["Stage"].unique())


# Only keep columns that we need
df = df[["Epoch", "Stage", "HR"]]

# Clean data, source: https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_dataset(df)

print("\n\nLabel-encoded dataframe:\n")
print(df)


# Machine Learning!
# TODO: Construct ???
train, test = train_test_split(df, test_size = 0.3)
print(train.shape)
print(test.shape)

train_X = train[["Epoch", "HR"]] # Input
train_Y = train.Stage # Output

test_X = test[["Epoch", "HR"]]

test_Y = test.Stage
# print("Test X and Test Y")
# print(test_X)
# print(test_Y)

# print(train_X.head())
# print(test_Y.head())

# SVM
model = svm.SVC(gamma="scale")

# model.fit(train_X.values, train_Y.values)
# prediction = model.predict(test_X.values)
# print("The accuracy of SVM is: ", metrics.accuracy_score(prediction, test_Y))

model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print("The accuracy of SVM is: ", metrics.accuracy_score(prediction, test_Y))

model = LogisticRegression(solver='lbfgs', multi_class="auto")
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print("The accuracy of Logistic Regression is: ", metrics.accuracy_score(prediction, test_Y))

model = KNeighborsClassifier(n_neighbors=3) # this examines 3 neighbors for putting the data into class
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print('The accuracy of KNN is: ', metrics.accuracy_score(prediction, test_Y))

model = DecisionTreeRegressor()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print("The MAE of Decision Tree Regressor is: ", metrics.mean_absolute_error(prediction, test_Y))

model = RandomForestRegressor(n_estimators = 600, random_state = 0)
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print("The MAE of SGDRegressor is: ", metrics.mean_absolute_error(prediction, test_Y))

model = svm.SVR(gamma="scale")
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print("The MAE of SVR is: ", metrics.mean_absolute_error(prediction, test_Y))

model = SGDRegressor()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print("The MAE of SGDRegressor is: ", metrics.mean_absolute_error(prediction, test_Y))