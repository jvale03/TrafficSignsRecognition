from PIL import Image
import numpy as np
from keras.models import load_model


classes = { 1:'Speed limit (20km/h)',
           2:'Speed limit (30km/h)',
           3:'Speed limit (50km/h)',
           4:'Speed limit (60km/h)',
           5:'Speed limit (70km/h)',
           6:'Speed limit (80km/h)',
           7:'End of speed limit (80km/h)',
           8:'Speed limit (100km/h)',
           9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing vehicle with a weight greater than 3.5 tons' }


def classify_img(img_path):
    image = None

    print("\033[32mLoading model...\033[m")
    try: 
        model = load_model("../Model/TrafficClassifier.keras")
        print("\033[32mModel loaded!\033[m")

    except Exception as e:
        print(f"\033[31mError: {e}")
        return None

    try:
        image = Image.open(img_path).convert('RGB')
        image = image.resize((30, 30))
        image = np.expand_dims(image, axis=0) 
        image = np.array(image) / 255.0
    except Exception as e:
        print(f"\033[31mError loading img: {e}")
        return None
    
    pred = model.predict(image)
    pred_classes = np.argmax(pred, axis=1).item()
    sign = classes[pred_classes+1]

    return sign 


    
