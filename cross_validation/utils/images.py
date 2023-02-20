import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from numba import jit, cuda

from skimage.io import imread_collection

from utils.pre_processed import equalizacao, filtro_mediana, otsu, segmentar

class Images:
    def __init__(self, metadata):
        self.path = metadata['image'].to_list()
        self.imgs = []
        self.labels = metadata['label'].to_numpy()
        self.imgs_normalized = []
        self.imgs_equalized = []
        self.imgs_processed = []
        self.masks = []
        self.imgs_segmented = []
        self.feats = {}
        self.load()
    

    def load(self):
        self.imgs = imread_collection(self.path)

        saudaveis = len(self.labels[self.labels == 0])
        infectadas = len(self.labels[self.labels == 1])
        
        print(f'{len(self.imgs)} imagens carregadas')
        print(f'{infectadas} infectadas')
        print(f'{saudaveis} saudáveis')

    def normalize(self):
        for img in tqdm_notebook(self.imgs):
            img_normalized = img.astype('float32') / 255.0
            self.imgs_normalized.append(img_normalized)


    def equalize(self):
        for img in tqdm_notebook(self.imgs_normalized):
            img_equalized = equalizacao(img)
            self.imgs_equalized.append(img_equalized)
            
    def filter_median(self):
        for img in tqdm_notebook(self.imgs_equalized):
            img_processed = filtro_mediana(img)
            self.imgs_processed.append(img_processed)


    def segment(self):
        for img in tqdm_notebook(self.imgs_processed):
            mask = otsu(img)
            img_segmented = segmentar(img, mask)
            self.masks.append(mask)
            self.imgs_segmented.append(img_segmented)


    
