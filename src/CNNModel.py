import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.metrics import accuracy_score
from keras import Input
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from DataExploration import data_exploration


X_t1, X_t2, y_t1, y_t2 = data_exploration()


model = Sequential()

def model_builder():
    global model
    try:
        model.add(Input(shape=X_t1.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
    except:
        print("Error building model!")

def model_compilation():
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    eps = 15
    model.fit(X_t1, y_t1, batch_size=32, epochs=eps, validation_data=(X_t2, y_t2))

def accuracy_test():
    print("ola")
    y_test = pd.read_csv("../DataSet/Test.csv")
    labels = y_test["ClassId"].values
    images = y_test["Path"].values
    data = []

    for img in images:
        try:
            image = Image.open(img)
            image = image.resize((30,30))
            data.append(np.array(image))
        except: 
                print("Error loading image") 

    X_test = np.array(data) 

    # pred = model.predict_classes(X_test)
    # print(accuracy_score(labels,pred))

    # model.save("../Model/TrafficClassifier.h5")