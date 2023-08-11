import os
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import sklearn
import lifelines
import shap

from util import *
from public_tests import *



# This sets a common size for all the figures we will draw.
#plt.rcParams['figure.figsize'] = [10, 7]

path = "./"
model = load_C3M3_model(path)

# IMAGE_DIR = 'data/nih_new/images-small/'
# df = pd.read_csv('data/nih_new/train-small.csv')
# im_path = IMAGE_DIR + '00025288_001.png' 
# x = load_image(im_path, df, preprocess=False)
# plt.imshow(x, cmap = 'gray')
# plt.show()

# functions.py

# functions.py

def process_image(input_filename):
    # Construct the full path to the input image in the dataset directory
    # input_image_path = os.path.join(IMAGE_DIR, input_filename)
    
    # Simulated processing logic: Just returning the input image path
    return input_filename

