import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from scipy.stats import skew, kurtosis, entropy
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm_notebook

def extract_features(descritor, imgs):
        feats = []

        
        if descritor == 'GLCM':
            for img in tqdm_notebook(imgs):
                feats.append(descritor_glcm(img))
        
        elif descritor == 'HOG':
            for img in tqdm_notebook(imgs):
                feats.append(descritor_hog(img))

        else:
            raise ValueError('Descritor não implementado')
        

        return np.array(feats)


def descritor_glcm(imagem):
    features = []

    distance = 2

    for d in range(imagem.shape[2]):
        matrix0 = graycomatrix((imagem[:,:,d]*255).astype(np.uint8), [distance], [0],normed=True)
        matrix1 = graycomatrix((imagem[:,:,d]*255).astype(np.uint8), [distance], [np.pi/4],normed=True)
        matrix2 = graycomatrix((imagem[:,:,d]*255).astype(np.uint8), [distance], [np.pi/2],normed=True)
        matrix3 = graycomatrix((imagem[:,:,d]*255).astype(np.uint8), [distance], [3*np.pi/4],normed=True)
        matrix = (matrix0+matrix1+matrix2+matrix3)/4 

        features.append(graycoprops(matrix,'contrast'))
        features.append(graycoprops(matrix,'dissimilarity'))
        features.append(graycoprops(matrix,'homogeneity'))
        features.append(graycoprops(matrix,'energy'))
        features.append(graycoprops(matrix,'correlation'))
        features.append(graycoprops(matrix,'ASM'))
    

    return np.array(features).flatten()
        

def descritor_hog(imagem):
    features = []
    for d in range(imagem.shape[2]):
        hist = hog((imagem[:,:,d]*255).astype(np.uint8))
        features.append(hist.mean())
        features.append(hist.var())
        features.append(skew(hist))
        features.append(kurtosis(hist))
        features.append(entropy(hist))
        features.append(np.sum(np.power(hist, 2)))
        
    return np.array(features).flatten()


def pca(x, components=2):
    reducer = PCA(n_components=components)
    return reducer.fit_transform(x)
