from PIL import Image
import numpy as np
from keras.models import load_model
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing vehicle with a weight greater than 3.5 tons'
}


def my_load_model(model_path="../Model/TrafficClassifier.keras"):
    print("\033[32mLoading model...\033[m")
    try:
        model = load_model(model_path)
        print("\033[32mModel loaded!\033[m")
        return model
    except (OSError, ValueError, Exception) as e:
        logging.error(f"\033[31mError 'my_load_model': {e}\033[m")
        return None


def preprocess_image(img_path):
    try:
        with Image.open(img_path).convert('RGB') as img:
            img = img.resize((30, 30))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
    except (OSError, ValueError, Exception) as e:
        logging.error(f"\033[31mError loading image: {e} in 'preprocess_image'\033[m")
        return None


def classify_img(model, img_path):
    if model is None:
        logging.warning("\033[31mModel is not loaded. Cannot classify image.\033[m")
        return None

    image = preprocess_image(img_path)
    if image is None:
        return None

    pred = model.predict(image)
    pred_classes = np.argmax(pred, axis=1).item()
    prob = np.max(pred)

    if prob < 0.6:  # Accept only if the model is sure about is answer
        return "Undefined"

    if pred_classes + 1 not in classes:
        logging.warning(f"\033[31mPredicted class {pred_classes + 1} not in defined classes.\033[m")
        return "Undefined"

    sign = classes[pred_classes + 1]
    return sign
