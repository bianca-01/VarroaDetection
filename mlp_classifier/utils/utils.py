import pickle
import json
from skimage.io import imread
from skimage.color import rgb2gray
import cv2

def save(content, name):
    try:
        with open(name, 'wb') as f:
            pickle.dump(content, f)
        print('Saved!')
    except Exception as e:
        print(e)

def read_json(path):
    with open(path) as json_file:
        paths = json.load(json_file)
    return paths

def read_img(path): #opencv
    img = cv2.imread(path,0)
    return img

def read_img2(path): #skimage
    img = imread(path)
    img = rgb2gray(img)
    return img

def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except:
        print('File not found')
        return None  