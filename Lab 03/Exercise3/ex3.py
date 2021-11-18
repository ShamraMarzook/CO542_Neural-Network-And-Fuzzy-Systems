# E/16/232

import os
from skimage import io
import numpy as np
from sklearn.linear_model import Perceptron

MAIN_PATH_TRAIN = r"C:\Users\user\Desktop\CO542\Lab3\Exercise3\classpics\train"
MAIN_PATH_TEST = r"C:\Users\user\Desktop\CO542\Lab3\Exercise3\classpics\test"
SUBSETS = ["brown_hair", "black_hair"]

features_train = []
target_train = []
for subset in SUBSETS:
    full_path = os.path.join(MAIN_PATH_TRAIN, subset)
    full_path_listdir = os.listdir(full_path)
    for img_name in full_path_listdir:
        img_path = os.path.join(full_path, img_name)
        image = io.imread(img_path)
        image = (np.array(image.ravel()))
        image = np.array(image)
        features_train.append(image)
        target_train.append(subset)

features_test = []
target_test = []
for subset in SUBSETS:
    full_path = os.path.join(MAIN_PATH_TEST, subset)
    full_path_listdir = os.listdir(full_path)
    for img_name in full_path_listdir:
        img_path = os.path.join(full_path, img_name)
        image = io.imread(img_path)
        image = (np.array(image.ravel()))
        image = np.array(image)
        features_test.append(image)
        target_test.append(subset)

# Building a perceptron model and train the data
classifier = Perceptron(max_iter=40)
classifier.fit(np.array(features_train), np.array(target_train))
print(classifier.score(np.array(features_train), np.array(target_train)))
predictions = classifier.predict(np.array(features_test))

print('Prediction:')
print(predictions)
print('Actual:')
print(np.array(target_test))
