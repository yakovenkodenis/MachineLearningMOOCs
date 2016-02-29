import numpy as np
from skimage.io import imread, imshow
from skimage import img_as_float
from sklearn.cluster import KMeans



def mse(I, K, m, n):
    divisor = 1 / (m * n * 3)
    result = 0
    for i in range(0, m * n):
        for j in range(0, 3):
            result += (I[i][j] - K[i][j]) ** 2
    return divisor * result
    
    
def psnr(I, n_mse):
    max = np.max(I)
    return 10 * np.log10((max ** 2) / n_mse)
    

parrots = imread('parrots.jpg')

img_floats_3d = img_as_float(parrots)
img_vec = np.ndarray.flatten(img_floats_3d)
img_mat = np.reshape(img_vec, (474 * 713, 3))



clf = KMeans(n_clusters=10, init='k-means++', random_state=241)
clf.fit(img_mat)

labels = clf.labels_

color_avgs = []

for cluster in range(0, len(set(labels))):
    avg_r = 0
    avg_g = 0
    avg_b = 0
    rgb_count = 0
    for i in range(0, len(labels)):
        if labels[i] == cluster:
            avg_r += img_mat[i][0]
            avg_g += img_mat[i][1]
            avg_b += img_mat[i][2]
            rgb_count += 1
    color_avgs.append((avg_r / rgb_count,
                       avg_g / rgb_count,
                       avg_b / rgb_count))


img_mat_new = []

for label in labels:
    img_mat_new.append(color_avgs[label])
    
img_new_3d = np.reshape(img_mat_new, (474, 713, 3))

imshow(img_floats_3d)
imshow(img_new_3d)

n_mse = mse(img_mat, img_mat_new, 474, 713)
n_psnr = psnr(img_mat, n_mse)

print(n_psnr)
