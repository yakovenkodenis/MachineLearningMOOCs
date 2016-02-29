import pandas
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor


data = pandas.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(
    lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))


target = np.array(data[data.columns[-1]])
features = np.array(data[data.columns[:-1]])


kfold = KFold(n=len(data), n_folds=5, shuffle=True, random_state=1)

score_list = []
for t in range(1, 51):
    forest = RandomForestRegressor(n_estimators=t, random_state=1)
    score = cross_val_score(forest, features,
                            cv=kfold, y=target,
                            scoring='r2').mean()

    if score > 0.52:
        print(t)
        break
