import numpy as np
import pandas as pd
from PIL import Image
import logging
from sklearn.metrics import accuracy_score
from keras import Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from DataExploration import data_exploration
import os

# Configuração do logging
logging.basicConfig(level=logging.INFO)

def model_builder():
    X_t1, X_t2, y_t1, y_t2 = data_exploration()
    
    model = Sequential()
    
    try:
        model.add(Input(shape=X_t1.shape[1:]))

        # Bloco 1
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        # Bloco 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        # Bloco 3
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))

    except Exception as e:
        logging.error(f"\033[31mError building model: {e}\n'model_builder'\033[m")
        return None, None, None, None

    return model, X_t1, X_t2, y_t1, y_t2

def model_compilation(model, X_t1, y_t1, X_t2, y_t2, epochs=11, batch_size=32):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_t1, y_t1, batch_size=batch_size, epochs=epochs, validation_data=(X_t2, y_t2))

def accuracy_test(model):
    y_test = pd.read_csv("../DataSet/Test.csv")
    labels = y_test["ClassId"].values
    images = y_test["Path"].values
    data = []

    for img in images:
        img_path = os.path.join("../DataSet/", img)
        try:
            with Image.open(img_path) as image:
                image = image.resize((30, 30))
                img_array = np.array(image) / 255.0
                data.append(img_array)
        except Exception as e:
            logging.error(f"\033[31mError loading image: {e}\n'accuracy_test'\033[m")
            continue

    X_test = np.array(data)
    pred = model.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)

    accuracy = accuracy_score(labels, pred_classes)
    print(f"Accuracy: {accuracy}")

    save_model(model)

def save_model(model):
    logging.info("\033[32mSaving model...\033[m")
    try:
        model.save("../Model/TrafficClassifier.keras")
        print("\033[32mModel saved successfully.\033[m")
    except Exception as e:
        logging.error(f"\033[31mError saving model: {e}\033[m")

# def save_model(model):
#     logging.info("Saving model...")
#     try:
#         model.save("TrafficClassifier.keras")
#         print("Model saved successfully.")
#         files.download("TrafficClassifier.keras")
#         print("Model downloaded successfully.")
#     except Exception as e:
#         logging.error(f"Error saving model: {e}")

def main():
    model, X_t1, X_t2, y_t1, y_t2 = model_builder()
    if model is not None:
        model_compilation(model, X_t1, y_t1, X_t2, y_t2)
        accuracy_test(model)

if __name__ == "__main__":
    main()

