import numpy as np
import pandas as pd
from PIL import Image
import logging
from sklearn.metrics import accuracy_score
from keras import Input
from DataExploration import data_exploration
import os
import pickle
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Add, Concatenate, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Configuração do logging
logging.basicConfig(level=logging.INFO)


def model_builder():
    X_t1, X_t2, y_t1, y_t2 = data_exploration()
    model = X_t1,X_t2,y_t1,y_t2
    # Guardar o modelo em bytes
    with open('../Model/NP_Images.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    try:
        input_layer = Input(shape=X_t1.shape[1:])

        # Bloco 1 - detecção multiescala
        conv1_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
        conv1_3 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
        conv1_5 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_layer)

        # Concatenar as saídas das convoluções
        concat_1 = Concatenate()([conv1_1, conv1_3, conv1_5])
        bn_1 = BatchNormalization()(concat_1)
        pool_1 = MaxPool2D(pool_size=(2, 2))(bn_1)
        dropout_1 = Dropout(0.25)(pool_1)

        # Bloco 2 - usando separable convolutions para mais eficiência
        conv2_1 = SeparableConv2D(64, (3, 3), padding='same', activation='relu')(dropout_1)
        conv2_2 = SeparableConv2D(128, (3, 3), padding='same', activation='relu')(conv2_1)
        bn_2 = BatchNormalization()(conv2_2)
        pool_2 = MaxPool2D(pool_size=(2, 2))(bn_2)
        dropout_2 = Dropout(0.25)(pool_2)

        # Bloco 3 - Skip Connection para preservar informações
        conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(dropout_2)
        # Ajuste da forma de dropout_2
        adjusted_dropout_2 = Conv2D(256, (1, 1), padding='same')(dropout_2)  # Ajusta a forma
        skip_conn = Add()([conv3_1, adjusted_dropout_2])  # Skip Connection
        bn_3 = BatchNormalization()(skip_conn)
        pool_3 = MaxPool2D(pool_size=(2, 2))(bn_3)
        dropout_3 = Dropout(0.5)(pool_3)

        # Bloco 4
        conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(dropout_3)
        bn_4 = BatchNormalization()(conv4_1)
        pool_4 = MaxPool2D(pool_size=(2, 2))(bn_4)
        dropout_4 = Dropout(0.5)(pool_4)

        # Bloco 5
        conv5_1 = Conv2D(1024, (3, 3), padding='same', activation='relu')(dropout_4)
        bn_5 = BatchNormalization()(conv5_1)
        global_avg_pool_5 = GlobalAveragePooling2D()(bn_5)
        dropout_5 = Dropout(0.5)(global_avg_pool_5)

        # Classificação final
        dense_1 = Dense(1024, activation='relu')(dropout_5)
        dropout_6 = Dropout(0.5)(dense_1)
        output_layer = Dense(43, activation='softmax')(dropout_6)

        model = Model(inputs=input_layer, outputs=output_layer)

    except Exception as e:
        logging.error(f"\033[31mError building model: {e}\n'model_builder'\033[m")
        return None, None, None, None
    
    save_model(model,"../Model/ModelBuilder.keras")

    return model, X_t1, X_t2, y_t1, y_t2

def model_compilation(model, X_t1, X_t2, y_t1, y_t2, epochs=11, batch_size=64):
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

    save_model(model,"../Model/TrafficClassifier.keras")

def save_model(model,path):
    print("\033[32mSaving model...\033[m")
    try:
        model.save(path)
        print("\033[32mModel saved successfully.\033[m")
    except Exception as e:
        logging.error(f"\033[31mError saving model: {e}\033[m")

def main():
    model, X_t1, X_t2, y_t1, y_t2 = model_builder()
    if model is not None:
        model_compilation(model, X_t1, y_t1, X_t2, y_t2)
        accuracy_test(model)

if __name__ == "__main__":
    main()

