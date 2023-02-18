import numpy as np
from skimage.filters import threshold_otsu

def alargamento(img, k, e):
    imgAl = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgAl[i][j] = 1/(1 + (k/img[i][j])**e)
    return imgAl

def otsu(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary


