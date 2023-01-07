import numpy as np
from skimage.exposure import histogram
from scipy.stats import skew, kurtosis, entropy
from tqdm.notebook import tqdm_notebook
from utils.utils import read_img

def histograma(img):
    hist = histogram(img, nbins=256)
    return hist[0]

def media(hist):
    return np.mean(hist)


def variancia(hist):
    return np.var(hist)


def skewness(hist):
    return skew(hist)


def entropia(hist):
    return entropy(hist)


def energia(hist):
    return np.sum(np.power(hist, 2))


def get_kurtosis(hist):
    return kurtosis(hist)


def descritor_histograma(img):
    if type(img) == str:
        img = read_img(img)
    hist = histograma(img)
    feats = [media(hist), 
            variancia(hist), 
            skewness(hist), 
            entropia(hist), 
            energia(hist), 
            get_kurtosis(hist)]
    return feats


def extract_features(imgs, path=None, op=1):
    features = []

    if op == 0: #dataframe
        for i in tqdm_notebook(imgs.index):
            descritor = descritor_histograma(str(path+imgs['image'][i]))
            features.append(descritor)

    elif op == 1: #lista
        for img in tqdm_notebook(imgs):
            descritor = descritor_histograma(img)
            features.append(descritor)

    else:
        print('opcao invalida')

    return features



