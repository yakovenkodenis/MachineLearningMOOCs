import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


train_data = pandas.read_csv('perceptron-train.csv', header=None)
test_data = pandas.read_csv('perceptron-test.csv', header=None)

train_target = train_data[[0]]
train_features = train_data[train_data.columns[1:3]]
test_target = test_data[[0]]
test_features = test_data[test_data.columns[1:3]]

train_X = np.array(train_features)
train_y = np.array(train_target)
test_X = np.array(test_features)
test_y = np.array(test_target)

clf = Perceptron(random_state=241)

clf.fit(train_X, train_y)
predictions = clf.predict(test_X)

accuracy = accuracy_score(test_target, predictions)

scaler = StandardScaler()

train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

clf.fit(train_X_scaled, train_y)
predictions = clf.predict(test_X_scaled)

accuracy_scaled = accuracy_score(test_target, predictions)

print((accuracy_scaled - accuracy))
