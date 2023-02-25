import numpy as np
from tqdm.notebook import tqdm_notebook
from skimage.io import imread
from utils.pre_processed import equalizacao, otsu, segmentar, filtro_mediana
from utils.feature_extraction import extract_features, pca

class Images:
    def __init__(self, metadata):
        self.path = metadata['image'].to_list()
        self.imgs = []
        self.labels = metadata['label'].to_numpy()
        self.imgs_normalized = []
        self.imgs_equalized = []
        self.masks = []
        self.imgs_segmented = []
        self.feats = {}
        self.pca = {}
        self.load()
    

    def load(self):
        self.imgs = [imread(path) for path in self.path]
        
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
            

    def segment(self):
        for img in tqdm_notebook(self.imgs_equalized):
            mask = otsu(img)
            img_segmented = segmentar(img, mask)
            self.masks.append(mask)
            self.imgs_segmented.append(img_segmented)

    
    def extraction(self, descritor='GLCM', op=1):
        if op == 1:
            imgs = self.imgs_segmented
            index = f'{descritor}_segmented'

        elif op == 2:
            imgs = self.imgs_equalized
            index = f'{descritor}_equalized'
        
        self.feats[index] = extract_features(descritor, imgs)


    def run_pca(self):
        components = [3, 5, 7, 9]
        for comp in tqdm_notebook(components):
            for key in tqdm_notebook(self.feats.keys()):
                self.pca[f'{key}_pca_{comp}'] = pca(self.feats[key], comp)

    
    


    
