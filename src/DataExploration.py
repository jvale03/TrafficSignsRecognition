import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

classes = 43

# Get the project's path
cur_path = os.getcwd()

def image_labels():
    data = []
    labels = []
    for i in range(classes):
        # Iterate over each class
        path = os.path.join(cur_path, '../DataSet/Train', str(i))

        if not os.path.exists(path):
            logging.warning(f"\033[31mPath does not exist: {path}\033[m")
            continue

        images = os.listdir(path)

        # Iterate over each image to resize and convert to an np.array
        for image_name in images:
            image_path = os.path.join(path, image_name)

            try:
                with Image.open(image_path) as img:
                    img = img.resize((30, 30))
                    img_array = np.array(img)

                    if img_array.shape == (30, 30, 3):  # Check if the image is the correct shape
                        img_array = img_array / 255.0
                        data.append(img_array)
                        labels.append(i)

            except (OSError, ValueError, Exception) as e:
                logging.error(f"\033[31mError loading image {image_name}: {e}\033[m")

    return np.array(data), np.array(labels)

def data_split(data, labels):
    X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Labels -> one hot encoding
    y_t1 = to_categorical(y_t1, num_classes=classes)
    y_t2 = to_categorical(y_t2, num_classes=classes)

    return X_t1, X_t2, y_t1, y_t2

def data_exploration():
    data, labels = image_labels()
    return data_split(data, labels)
