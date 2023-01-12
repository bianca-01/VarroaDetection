import numpy as np
from tqdm.notebook import tqdm_notebook
from utils.utils import read_img, read_img2

def alargamento(img, k, e):
    imgAl = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgAl[i][j] = 1/(1 + (k/img[i][j])**e)
    return imgAl


def negativo(img):
    imgNeg = 255 - img
    return imgNeg


def logaritmico(img, c):
    imgLog = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgLog[i][j] = c * np.log10(1+img[i][j])
    return imgLog


def potencia(img, c, gama):
    imgP = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgP[i][j] = c * (img[i][j]**gama)
    return imgP

def pre_processamento(path, metadata, funcao, parametros=None):
    imgs = []

    for i in tqdm_notebook(metadata.index):
        img = read_img(path+metadata['image'][i])

        
        if parametros is None:
            imgPro = funcao(img)
        elif len(parametros) == 1:
            imgPro = funcao(img, parametros[0])
        elif len(parametros) == 2:
            imgPro = funcao(img, parametros[0], parametros[1])

        imgs.append(imgPro)

    return np.array(imgs)