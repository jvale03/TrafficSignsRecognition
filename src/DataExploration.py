import numpy as np
from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 


data = []
labels = []
classes = 43

# get the projects path
cur_path = os.getcwd()


# on "train" folder, each subfolder represents a class
def image_labels():
    global data, labels
    for i in range(classes):
        # iterate over each class
        path = os.path.join(cur_path,'../DataSet/Train', str(i)) 
        images = os.listdir(path)

        # iterate over each image to resize them and convert in an np.array
        for a in images:
            try:
                image = Image.open(path + '/' + a) 
                image = image.resize((30,30)) 
                image = np.array(image)
                if image.shape == (30, 30, 3):  # Check that image is the right shape
                    image = np.array(image) / 255.0
                    data.append(image) 
                    labels.append(i) 
            except Exception as e:
                print(f"\033[31mError loading img: {e}\033[m")
                return None

    # convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)


def data_split():
    X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.25, random_state=42)

    # labels -> one hot encoding
    y_t1 = to_categorical(y_t1, 43)
    y_t2 = to_categorical(y_t2, 43) 

    return X_t1, X_t2, y_t1, y_t2

def data_exploration():
    image_labels()
    return data_split()

