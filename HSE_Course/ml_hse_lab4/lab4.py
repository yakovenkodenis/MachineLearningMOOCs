import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score


boston = load_boston()
boston_features = scale(boston.data, with_mean=True, with_std=True)

kfold = KFold(n=len(boston.target), n_folds=5, shuffle=True, random_state=42)

opt_p = 0.0
opt_score = float('-inf')
for pp in np.linspace(1, 10, num=200):
    neighbour = KNeighborsRegressor(n_neighbors=5, weights='distance', p=pp)
    scores = cross_val_score(neighbour, boston_features,
                             cv=kfold, y=boston.target,
                             scoring='mean_squared_error')
    mean_score = np.mean(scores)
    print("P: %s, SCORE: %s\n" % (pp, mean_score))
    if mean_score >= opt_score:
        opt_score = mean_score
        opt_p = pp

print
print
print("P: %s, SCORE: %s\n" % (opt_p, opt_score))
