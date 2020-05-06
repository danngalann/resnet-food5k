from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import imutils
from imutils import paths
import cv2
import random

# Load a random image
imagePaths = list(paths.list_images("dataset/"))

image = cv2.imread(random.choice(imagePaths[1000:]))
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (224,224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Load model
model = tf.keras.models.load_model("model")

# Classify image
(notFood, food) = model.predict(image)[0]

# build the label
label = "Food" if food > notFood else "Not Food"
proba = food if food > notFood else notFood
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)