from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# Initialize variables
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# Initialize data and labels
print("Loading images...")
data = []
labels = []

# Load data and labels
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # Load image
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224,224))
    image = img_to_array(image)
    data.append(image)

    # Load label
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "food" else 0
    labels.append(label)

print(labels)