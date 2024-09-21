import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from keras import Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from DataExploration import data_exploration


X_t1, X_t2, y_t1, y_t2 = None, None, None, None

model = Sequential()

def model_builder():
    global model, X_t1, X_t2, y_t1, y_t2
    X_t1, X_t2, y_t1, y_t2 = data_exploration()
    try:
        model.add(Input(shape=X_t1.shape[1:]))
        # block 1
        model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        # block 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        # block 3
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
    except:
        print("\033[31mError building model!\033[m")

def model_compilation():
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    eps = 11
    model.fit(X_t1, y_t1, batch_size=32, epochs=eps, validation_data=(X_t2, y_t2))

def accuracy_test():
    y_test = pd.read_csv("../DataSet/Test.csv")
    labels = y_test["ClassId"].values
    images = y_test["Path"].values
    data = []

    for img in images:
        img = "../DataSet/"+img
        try:
            image = Image.open(img)
            image = image.resize((30,30))
            image = np.array(image) / 255.0
            data.append(image)
        except Exception as e:
            print(f"\033[31mError loading img: {e}\033[m")
            return None

    X_test = np.array(data) 

    pred = model.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)

    print(accuracy_score(labels,pred_classes))

    save_model()

def save_model():
    print("\033[32mSaving model...\033[m")
    try: 
        model.save("../Model/TrafficClassifier.keras")
        print("\033[32mModel saved...\033[m")
        
    except Exception as e:
        print(f"\033[31mError: {e}\033[m")
