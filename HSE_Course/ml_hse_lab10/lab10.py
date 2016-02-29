import re
import pandas
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


data_train = pandas.read_csv('salary-train.csv')
data_train['FullDescription'] = data_train['FullDescription'].apply(
    lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))

target = data_train['SalaryNormalized']
features = data_train[['FullDescription',
                       'LocationNormalized',
                       'ContractTime']]

features['LocationNormalized'].fillna('nan', inplace=True)
features['ContractTime'].fillna('nan', inplace=True)

TfIdf = TfidfVectorizer(min_df=5)

full_description = TfIdf.fit_transform(features['FullDescription'])


enc = DictVectorizer()
X_train_categ = enc.fit_transform(
    features[['LocationNormalized', 'ContractTime']]
    .to_dict('records'))

final_data_train = hstack([full_description, X_train_categ])

clf = Ridge(alpha=1.0)

clf.fit(final_data_train, target)


test_data = pandas.read_csv('salary-test-mini.csv')

test_features = test_data[['FullDescription',
                           'LocationNormalized',
                           'ContractTime']]

test_features['LocationNormalized'].fillna('nan', inplace=True)
test_features['ContractTime'].fillna('nan', inplace=True)


test_full_desc = TfIdf.transform(test_features['FullDescription'])

X_test_categ = enc.transform(
    test_features[['LocationNormalized', 'ContractTime']]
    .to_dict('records'))

final_data_test = hstack([test_full_desc, X_test_categ])

predictions = clf.predict(final_data_test)

print(np.round(predictions, 2))
