import numpy as np
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.exposure import equalize_hist

def equalizacao(imagem):
    img_eq = np.zeros_like(imagem)

    if len(imagem.shape) == 3:
        for d in range(imagem.shape[2]):
            img_eq[:,:,d] = equalize_hist(imagem[:,:,d])

    else:
        img_eq = equalize_hist(imagem)

    return img_eq


def otsu(img):
    thresh = threshold_otsu(img[:,:,2])
    mask = img[:,:,2] > thresh
    return invert(mask)


def segmentar(imagem, mask):
    img_seg = imagem.copy()
    img_seg[~mask] = 0
    return img_seg
    

