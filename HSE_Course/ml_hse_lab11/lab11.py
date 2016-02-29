import pandas
import numpy as np
from sklearn.decomposition import PCA


close_prices = pandas.read_csv('close_prices.csv').drop('date', 1)

pca = PCA(n_components=10)
pca.fit(close_prices)

min_components = 0
tmp_sum = 0
for c in sorted(pca.explained_variance_ratio_, reverse=True):
    tmp_sum += c
    min_components += 1
    if tmp_sum >= 0.9:
        break


new_data = pca.transform(close_prices)

dow_jones = np.array(pandas.read_csv('djia_index.csv')['^DJI'])


first_component = new_data[::, 0]
pearson_corr = np.corrcoef(first_component, dow_jones)


max_var_ratio = np.max(pca.explained_variance_ratio_)

comps = pca.components_

comp_res = (0, 0)
for i in range(0, 30):
    if np.abs(comps[0][i]) > comp_res[1]:
        comp_res = (i+1, comps[0][i])

print(comp_res)
