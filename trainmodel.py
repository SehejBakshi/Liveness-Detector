# USAGE
# python trainmodel.py --dataset dataset --model liveness.model --le le.pickle

import matplotlib
matplotlib.use("Agg")

from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
#from keras.models import model_from_json
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
               help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
                help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

init_lr = 1e-4
bs = 8
epochs = 50

print("Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0

print('data shape:', data.shape)
print('label shape:', len(labels))
#print(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

print("Compiling Model...")
opt = Adam(lr=init_lr, decay=init_lr/epochs)

model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("Training network for {} epochs...".format(epochs))

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=bs),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // bs,
            epochs=epochs)

tfjs_target_dir = 'C:/Python37/Projects/Liveness Detector'
tfjs.converters.save_keras_model(model, tfjs_target_dir)


print("Evaluating...")
predictions = model.predict(testX, batch_size=bs)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=le.classes_))

print("Serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
