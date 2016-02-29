import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix


newsgroups = datasets.fetch_20newsgroups(
    subset='all', categories=['alt.atheism', 'sci.space'])

data = newsgroups.data
target = newsgroups.target

TfIdf = TfidfVectorizer()
data_scaled = TfIdf.fit_transform(data)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
kfold = KFold(n=len(target), n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kfold)
gs.fit(data_scaled, target)

best_C = gs.best_params_['C']
clf = SVC(C=best_C, kernel='linear', random_state=241)

clf.fit(data_scaled, target)

coefs = clf.coef_


res_dict = dict()

print(type(coefs))

cx_coefs = coo_matrix(coefs)

print("======")

for i, j, v in zip(cx_coefs.row, cx_coefs.col, cx_coefs.data):
    res_dict[j] = abs(v)

# for val, (key1, key2) in coefs:
#     res_dict[key2] = abs(coefs[(key1, key2)])


sorted_indices = []

for ind in sorted(res_dict, key=res_dict.get, reverse=True):
    sorted_indices.append(ind)

words = TfIdf.get_feature_names()

result = []

for i in sorted_indices[:10]:
    result.append(words[i])

print(sorted(result))
