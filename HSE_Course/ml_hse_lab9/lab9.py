import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


classification_data = pandas.read_csv('classification.csv')

classification_array = np.array(classification_data)

TP, FP, FN, TN = 0, 0, 0, 0

for pair in classification_array:
    TP += pair[0] == pair[1] == 1
    TN += pair[0] == pair[1] == 0
    FP += pair[0] == 0 and pair[1] == 1
    FN += pair[0] == 1 and pair[1] == 0


accuracy = accuracy_score(classification_data['true'],
                          classification_data['pred'])
precision = precision_score(classification_data['true'],
                            classification_data['pred'])
recall = recall_score(classification_data['true'], classification_data['pred'])
f1 = f1_score(classification_data['true'], classification_data['pred'])


scores_data = pandas.read_csv('scores.csv')

target = scores_data['true']
logreg = scores_data['score_logreg']
svm = scores_data['score_svm']
knn = scores_data['score_knn']
tree = scores_data['score_tree']

roc_auc_logreg = roc_auc_score(target, logreg)
roc_auc_svm = roc_auc_score(target, svm)
roc_auc_knn = roc_auc_score(target, knn)
roc_auc_tree = roc_auc_score(target, tree)


def compute_max_precision(target, features):
    precision, recall, thresholds = precision_recall_curve(target, features)

    result = 0
    for i in range(0, len(precision)):
        if precision[i] > result and recall[i] >= 0.7:
            result = precision[i]

    return result

logreg_pr = compute_max_precision(target, logreg)
svm_pr = compute_max_precision(target, svm)
knn_pr = compute_max_precision(target, knn)
tree_pr = compute_max_precision(target, tree)
