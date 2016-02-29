import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np


init_data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data = init_data[['Pclass', 'Fare', 'Age', 'Sex']]
data = data[pandas.notnull(data['Age'])]
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})

target = init_data[pandas.notnull(init_data['Age'])]
target = target[['Survived']]

clf = DecisionTreeClassifier(random_state=241)
X = np.array(data)
y = np.array(target)

clf.fit(X, y)

importances = clf.feature_importances_
print(importances)

# [ 0.14551471  0.2933807   0.26059238  0.30051221]
