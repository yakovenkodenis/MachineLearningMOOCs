import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


data = pandas.read_csv('gbm-data.csv')

target = np.array(data[data.columns[0]])
features = np.array(data[data.columns[1:]])

X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.8,
                                                    random_state=241)


# for lr in [1, 0.5, 0.3, 0.2, 0.1]:
for lr in [0.2]:
    clf = GradientBoostingClassifier(n_estimators=250,
                                     verbose=True,
                                     random_state=241,
                                     learning_rate=lr)
    clf.fit(X_train, y_train)

    sigmoid_test_arr, sigmoid_train_arr = [], []

    train_pred = clf.staged_decision_function(X_train)
    test_pred = clf.staged_decision_function(X_test)

    test_pred_arr, train_pred_arr = [], []

    for i, val in enumerate(train_pred):
        sigmoid = 1 / (1 + np.exp(-val))
        train_pred_arr.append(log_loss(y_train, sigmoid))

    for i, val in enumerate(test_pred):
        sigmoid = 1 / (1 + np.exp(-val))
        test_pred_arr.append(log_loss(y_test, sigmoid))

    test_tuples, train_tuples = [], []

    i = 0
    for s in test_pred_arr:
        i += 1
        test_tuples.append((i, s))

    for t in sorted(test_tuples, key=lambda x: x[1]):
        print(t)

    # plt.figure()
    # plt.plot(test_pred_arr, 'r', linewidth=2)
    # plt.plot(train_pred_arr, 'g', linewidth=2)
    # plt.legend(['test', 'train'])


forest = RandomForestClassifier(n_estimators=37, random_state=241)

forest.fit(X_train, y_train)

probas = forest.predict_proba(X_test)

loss = log_loss(y_test, probas)
print(np.round(loss, 2))
