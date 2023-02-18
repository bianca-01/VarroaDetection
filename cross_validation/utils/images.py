import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook

from skimage.io import imread_collection

from utils.pre_processed import alargamento, otsu

class Images:
    def __init__(self, metadata):
        self.path = metadata['image'].to_list()
        self.imgs = []
        self.labels = metadata['label'].to_numpy()
        self.imgs_normalized = []
        self.imgs_processed = []
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


    def pre_process(self):
        for img in tqdm_notebook(self.imgs_normalized):
            img_processed = alargamento(img, 1, 2)
            self.imgs_processed.append(img_processed)


    def segment(self):
        for img in tqdm_notebook(self.imgs_processed):
            img_segmented = otsu(img)
            self.imgs_segmented.append(img_segmented)


    
