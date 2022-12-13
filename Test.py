# import tensorflow as tf
# import keras as k
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data, "\n")

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.1)
linear = linear_model.LinearRegression()


linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(f"The accuracy of the model is {acc * 100} %")


print("Coefficients: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
print("predictions", predictions)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i], sep=", ")
