import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

# Import dataset
data = pd.read_csv("student_mat_2173a47420.csv", sep=";")

# Extract necessary data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Our desired values to predict
predict = "G3"

X = np.array(data.drop([predict], axis=1))  # Features
Y = np.array(data[predict])  # Label

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

plot = "G2"
plt.scatter(data[plot], data["G3"], label="Student")
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()