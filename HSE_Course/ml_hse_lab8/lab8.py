import pandas
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score


def gradient_sum_part(w1, w2, k, X, y, X_column=0):
    l = len(y)
    res = k / l
    temp_res = 0
    for i in range(0, l):
        temp1 = y[0][i] * X[i][X_column]
        temp2 = 1 - 1 / (1 + np.exp(-y[0][i] * (w1*X[i][0] + w2*X[i][1])))
        temp_res += temp1 * temp2
    res *= temp_res
    return res


def gradient_step(w1, w2, k, X, y, C, L2=False):
    tmp_w1, tmp_w2 = w1, w2
    grad_sum_part_for_w1 = gradient_sum_part(tmp_w1, tmp_w2, k, X, y, 0)
    grad_sum_part_for_w2 = gradient_sum_part(tmp_w1, tmp_w2, k, X, y, 1)

    tp1 = w1 + grad_sum_part_for_w1 - k * C * w1 * L2
    tp2 = w2 + grad_sum_part_for_w2 - k * C * w2 * L2
    return (tp1, tp2)


def euclidean_distance_square(w1, w2):
    return euclidean_distance((w1, w2)) ** 2


def euclidean_distance(w_tuple):
    return np.linalg.norm(w_tuple)


def euclidean_distance_2d(w_tuple1, w_tuple2):
    return distance.euclidean(w_tuple1, w_tuple2)


def sigmoid_arr(w1, w2, X):
    result = []

    for i in range(0, len(X)):
        tmp = 1 / (1 + np.exp(-w1 * X[i][0] - w2 * X[i][1]))
        result.append(tmp)

    return result


def map_probabilistic_arr(arr):
    res = []
    for i in arr:
        if i == 1:
            res.append(1)
        else:
            res.append(0)

    return res


def logistic_regression_formula(w1, w2, X, y, k=0.1, C=10, L2=False):
    l = len(y)
    res = 1 / l
    tmp_res = 0

    for i in range(0, l):
        tmp_exp = np.exp(-y[0][i] * (w1*X[i][0] + w2*X[i][1]))
        tmp_res += np.log(1 + tmp_exp)
    res *= tmp_res

    L2_reg = (C / 2) * euclidean_distance_square(w1, w2)
    res += L2_reg * L2

    return res


def logistic_regression(X, y, k=0.1, C=10, L2=False):
    regression_res = float('inf')
    w1, w2 = 0, 0

    w_tuple_current, w_tuple_previous = (0, 0), (0, 0)

    for i in range(0, 10000):
        regression_res = logistic_regression_formula(w1, w2, X, y, k, C, L2)

        w_tuple_previous = (w1, w2)
        w_tuple_current = gradient_step(w1, w2, k, X, y, C, L2)

        if i % 100 == 0:
            print(w_tuple_current)

        if euclidean_distance_2d(w_tuple_current, w_tuple_previous) <= 1e-5:
            print("The result has been achieved on the iteration № %d" % i)
            print("Regression result: %f" % regression_res)

            return sigmoid_arr(w_tuple_current[0], w_tuple_current[1], X)

        w1 = w_tuple_current[0]
        w2 = w_tuple_current[1]

    print("The logistic regression algorithm has finished.")
    return None


init_data = pandas.read_csv('data-logistic.csv', header=None)
target = init_data[[0]]
features = np.array(init_data[init_data.columns[1:]])

sigmoid_array = logistic_regression(features, target, L2=False)
sigmoid_array_L2 = logistic_regression(features, target, L2=True)

y_arr = map_probabilistic_arr(target[0].tolist())

no_reg = roc_auc_score(y_arr, sigmoid_array)
L2_reg = roc_auc_score(y_arr, sigmoid_array_L2)

print("ROC_AUC no regularization: %f" % no_reg)
print("ROC_AUC L2 regularization: %f" % L2_reg)
