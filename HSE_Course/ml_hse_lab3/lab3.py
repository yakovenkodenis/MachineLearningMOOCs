import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale


init_data = pandas.read_csv('data.csv', header=None)
coll_list = init_data.columns.tolist()

classes = init_data[[0]]
features = init_data[coll_list[1:]]

kfold = KFold(n=len(init_data.index), n_folds=5, shuffle=True, random_state=42)


opt_k = 0
opt_score = 0.0
for k in range(1, 51):
    neighbour = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(neighbour, features, cv=kfold,
                            y=classes[0].values, scoring='accuracy')
    mean_score = np.mean(score)
    if mean_score > opt_score:
        opt_k = k
        opt_score = mean_score


print("K: %s, SCORE: %s\n" % (opt_k, opt_score))

scaled_features = scale(features, with_mean=True, with_std=True)

for k in range(1, 51):
    neighbour = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(neighbour, scaled_features, cv=kfold,
                            y=classes[0].values, scoring='accuracy')
    mean_score = np.mean(score)
    if mean_score > opt_score:
        opt_k = k
        opt_score = mean_score


print("K: %s, SCORE: %s\n" % (opt_k, opt_score))
