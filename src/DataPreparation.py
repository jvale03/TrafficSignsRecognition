from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import os
from datetime import datetime

db_path = os.path.join(os.getcwd(),"../DataSet/")

def imgs_augmentation_train():
    global db_path
    db_path = os.path.join(db_path,"Train")
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path,folder)
        if os.path.isdir(folder_path): # check if it is a directory
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path,img)
                try:
                    gen_images(img_path,folder_path)
                except Exception as e:
                    print(f"\033[31mError generating imgs: {e}")


def remove_img_augmentation():
    global db_path
    db_path = os.path.join(db_path,"Train")
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path,folder)
        if os.path.isdir(folder_path): # check if it is a directory
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path,img)
                if "image" in img:
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"\033[31mError removing img: {e}")
        
def gen_images(img_path,dir):
    datagen = ImageDataGenerator(
        rotation_range = 30,
        shear_range = 0.2,
        zoom_range = 0.2,
        brightness_range = (0.6, 1.4))
    
    img = load_img(img_path)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, ) + img_array.shape)

    i = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for batch in datagen.flow(img_array, batch_size = 1,
                          save_to_dir = dir, 
                          save_prefix =f'image{timestamp}{i}', save_format ='png'):
        i+=1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if i<5:
            break

def main():
    choice = 0
    while True:
        choice = input("1: Image augmentation\n2: Remove created images\n3: Cancel\nInput: ")
        if choice.isdigit():
            choice = int(choice)
            if choice > 2 or choice < 1:
                print("\033[31mInvalid!\033[m")
            else:
                break
        else:
            print("\033[31mInvalid!\033[m") 

    if choice == 1:
        print("\033[32mGenerating images...\033[m")
        try:
            imgs_augmentation_train()
            print("\033[32mOmages generated!\033[m")

        except Exception as e:
            print(f"\033[31mError generating imgs: {e}")
    elif choice == 2:
        print("\033[32mRemoving images...\033[m")
        try:
            remove_img_augmentation()
            print("\033[32mImages removed!\033[m")

        except Exception as e:
            print(f"\033[31mError removing imgs: {e}")
    else:
        return 0
main()