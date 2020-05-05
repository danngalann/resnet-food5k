from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import tensorflow as tf
import tensorflow_hub as hub
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# Initialize variables
EPOCHS = 2
INIT_LR = 1e-3
BS = 32
num_classes = 2

# Initialize data and labels
print("Loading images...")
data = []
labels = []

# Load data and labels
imagePaths = sorted(list(paths.list_images("dataset")))[:1000]
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

# Scale images to range [0,1] and convert to np array
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert labels to vectors
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)

# Construct image data generator
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# Build model
print("Building model...")
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])
model.build([None, 224, 224, 3])

# Compile model
print("Compiling model")
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("Training model...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# Save model
print("Saving model...")
model.save("model")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Food/Not Food")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")