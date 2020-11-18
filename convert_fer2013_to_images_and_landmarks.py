import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2

from skimage.feature import hog

import pdb
from PIL import Image, ImageOps

# initialization
image_height = 1280
image_width = 720
window_size = 24
window_step = 6
ONE_HOT_ENCODING = True
SAVE_IMAGES = False
GET_LANDMARKS = True
GET_HOG_FEATURES = False
GET_HOG_IMAGES = False
GET_HOG_WINDOWS_FEATURES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = 500
OUTPUT_FOLDER_NAME = "tjdata_features"
DATA_FOLDER_NAME= "data"

SELECTED_LABELS = [0,1]
print( str(len(SELECTED_LABELS)) + " expressions")

# loading Dlib predictor and preparing arrays:
print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
original_labels = [0, 1]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)

def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1)))
    return hog_windows


print( "importing csv file")
data = pd.read_csv('tjdata.csv')

for category in data['Usage'].unique():
    print( "converting set: " + category + "...")
    # create folder
    if not os.path.exists(category):
        try:
            os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
               pass
            else:
                raise
    
    # get samples and labels of the actual category
    category_data = data[data['Usage'] == category]
    samples = category_data['Num'].values
    labels = category_data['Class'].values
    
    # get images and extract features
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    for i in range(len(samples)):
        try:
            if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
                #image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                image_tmp = Image.open(DATA_FOLDER_NAME+'/'+category+'/'+category+np.array2string(samples[i]).zfill(4)+'.jpg')
                image_tmp = ImageOps.grayscale(image_tmp)
                image = np.array(image_tmp)
                images.append(image)
                if SAVE_IMAGES:
                    print('place holder')
                    #scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)
                if GET_HOG_WINDOWS_FEATURES:
                    features = sliding_hog_windows(image)
                    f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                            cells_per_block=(1, 1))
                    hog_features.append(features)
                    if GET_HOG_IMAGES:
                        hog_images.append(hog_image)
                elif GET_HOG_FEATURES:
                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                            cells_per_block=(1, 1))
                    hog_features.append(features)
                    if GET_HOG_IMAGES:
                        hog_images.append(hog_image)
                if GET_LANDMARKS:
                    #deprecated scipy.misc.imsave
                    #scipy.misc.imsave('temp.jpg', image)
                    image_tmp.save('tmp.jpg')
                    image2 = cv2.imread('temp.jpg')
                    face_rects = [dlib.rectangle(left=1, top=1, right=1280, bottom=720)]
                    face_landmarks = get_landmarks(image2, face_rects)
                    landmarks.append(face_landmarks)            
                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
                nb_images_per_label[get_new_label(labels[i])] += 1
        except Exception as e:
            print( "error in image: " + str(i) + " - " + str(e))

    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)
    if ONE_HOT_ENCODING:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
    else:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
    if GET_LANDMARKS:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/landmarks.npy', landmarks)
    if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_features.npy', hog_features)
        if GET_HOG_IMAGES:
            np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_images.npy', hog_images)
