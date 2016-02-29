import pandas
import numpy as np
from sklearn.svm import SVC


init_data = pandas.read_csv('svm-data.csv', header=None)
target = np.array(init_data[[0]])
features = np.array(init_data[init_data.columns[1:]])

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(features, target)

print(clf.support_)
