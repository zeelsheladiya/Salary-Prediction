import os

import numpy as np
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')
# print(data)

# write your code here
# X = data[["rating"]]
# y = data[["salary"]]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#
# linearModel = LinearRegression()
# linearModel.fit(X_train, y_train)
# y_pred = linearModel.predict(X_test)
#
# mean_absolute_percentage_error = float((mape(y_test, y_pred)).round(5))
# intercept = float((linearModel.intercept_.flatten()).round(5))
# slope = float((linearModel.coef_.flatten()).round(5))
#
# print(f"{intercept} {slope} {mean_absolute_percentage_error}")

# mape_arr = []
#
# for p in [2, 3, 4]:
#     # write your code here
#     X = data.drop(['salary'],axis=1).values
#     y = data[["salary"]].values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#
#     linearModel = LinearRegression()
#     linearModel.fit(X_train, y_train)
#     y_pred = linearModel.predict(X_test)
#
#     mean_absolute_percentage_error = float((mape(y_test, y_pred)).round(5))
#     # intercept = float((linearModel.intercept_.flatten()).round(5))
#     # slope = float((linearModel.coef_.flatten()).round(5))
#
#     mape_arr.append(mean_absolute_percentage_error)
#
# print(min(mape_arr))


# X = data.drop(['salary'], axis=1)
# y = data[["salary"]].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#
#
# linearModel = LinearRegression()
# linearModel.fit(X_train, y_train)
# # y_pred = linearModel.predict(X_test)
#
# # mean_absolute_percentage_error = float((mape(y_test, y_pred)).round(5))
# # intercept = float((linearModel.intercept_.flatten()).round(5))
# # slope = linearModel.coef_.flatten()
#
# print(*linearModel.coef_.flatten(), sep=", ")

# mape_list = []
# #
# # print(data.corr())
# X = data.drop(['salary'], axis=1)
# X1 = data.drop(['salary', "age", "experience"], axis=1)
# X2 = data.drop(["age", 'salary'], axis=1)
# # X3 = data.drop(["experience", "age"], axis=1) # it has 0 mape score
# X4 = data.drop(["experience", "salary"], axis=1)
# X5 = data.drop(["salary"], axis=1)
# # X6 = data.drop(["age"], axis=1) # it has 0 mape score
# # X7 = data.drop(["experience"], axis=1) # it has 0 mape score
#
# X_list = [X1, X2, X4, X5]
#
# y = data[["salary"]].values
#
# # print(X)
#
# for i in X_list:
#     X_train, X_test, y_train, y_test = train_test_split(i, y, test_size=0.3, random_state=100)
#
#     linearModel = LinearRegression()
#     linearModel.fit(X_train, y_train)
#     y_pred = linearModel.predict(X_test)
#
#     mean_absolute_percentage_error = float((mape(y_test, y_pred)).round(5))
#
#     mape_list.append(mean_absolute_percentage_error)
#
# print(min(mape_list))
# print(mape_list)


X = data.drop(['salary', "age", "experience"], axis=1)
y = data[["salary"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
y_pred = linearModel.predict(X_test)

y_pred1 = y_pred
y_pred2 = y_pred

y_pred1[y_pred1 < 0] = 0
y_pred2[y_pred2 < 0] = np.median(y_train)

mape1 = float((mape(y_test, y_pred1)).round(5))
mape2 = float((mape(y_test, y_pred2)).round(5))

print(min(mape1, mape2))