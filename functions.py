
import os
from keras.preprocessing import image
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import sklearn
import lifelines
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from util import *




# This sets a common size for all the figures we will draw.
plt.rcParams['figure.figsize'] = [10, 7]
path = "./"
IMAGE_DIR = 'data/nih_new/images-small/'
df = pd.read_csv('data/nih_new/train-small.csv')
np.random.seed(42)
model = load_C3M3_model(path) 

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
labels_to_show = ['Cardiomegaly', 'Mass', 'Edema']


def grad_cam(input_model, image, category_index, layer_name):

    cam = None

    output_with_batch_dim = input_model.output
    output_all_categories = output_with_batch_dim[0]
    y_c = output_all_categories[category_index]
    spatial_map_layer = input_model.get_layer(layer_name).output
    grads_l = K.gradients(y_c, spatial_map_layer)
    grads = grads_l[0]
    spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])
    spatial_map_val = spatial_map_all_dims[0]
    grads_val = grads_val_all_dims[0]
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.dot(spatial_map_val, weights)

    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()

    return cam


def compute_gradcam(model, img, mean, std, data_dir, df,
                    labels, selected_labels, layer_name='conv5_block16_concat'):
    img_path = data_dir + img
    preprocessed_input = load_image_normalize(img_path, mean, std)
    predictions = model.predict(preprocessed_input)

    highest_prob_label_index = np.argmax(predictions)
    highest_prob_label = labels[highest_prob_label_index]

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')

    gradcam = grad_cam(model, preprocessed_input, highest_prob_label_index, layer_name)

    plt.subplot(122)
    plt.title(highest_prob_label + ": " + str(round(predictions[0][highest_prob_label_index], 3)))
    plt.axis('off')
    plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')
    plt.imshow(gradcam, cmap='magma', alpha=min(0.5, predictions[0][highest_prob_label_index]))


            
image_name = '00004090_002.png'
mean, std = get_mean_std_per_batch(IMAGE_DIR, df)
compute_gradcam(model, image_name, mean, std, IMAGE_DIR, df, labels, labels_to_show)