from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
import os
from datetime import datetime
from PIL import Image

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
                    print(f"\033[31mError generating imgs: {e}\033[m")


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
                        print(f"\033[31mError removing img: {e}\033[m")
        

def png_converter():
    global db_path
    db_path = os.path.join(db_path,"Train")
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path,folder)
        if os.path.isdir(folder_path): # check if it is a directory
            for image in os.listdir(folder_path):
                if image.endswith(".jpg") or image.endswith(".jpeg"):
                    img_path = os.path.join(folder_path, image)
                    img = Image.open(img_path)
                    # Remover a extensão .jpg
                    nome_ficheiro = os.path.splitext(image)[0]
                    # Converter para .png e guardar
                    img.save(os.path.join(folder_path, nome_ficheiro + ".png"))
                    os.remove(img_path)


def gen_images(img_path,dir):
    datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.3,
        brightness_range = (0.4, 1.6))
    
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
        if i<6:
            break



def main():
    try:
        choice = 0
        while True:
            choice = input("1: Image augmentation\n2: Remove created images\n3: Convert to png\n4: Cancel\nInput: ")
            if choice.isdigit():
                choice = int(choice)
                if choice > 3 or choice < 1:
                    print("\033[31mInvalid!\033[m")
                else:
                    break
            else:
                print("\033[31mInvalid!\033[m") 

        if choice == 1:
            print("\033[32mGenerating images...\033[m")
            try:
                imgs_augmentation_train()
                print("\033[32mImages generated!\033[m")

            except Exception as e:
                print(f"\033[31mError generating imgs: {e}\033[m")

        elif choice == 2:
            print("\033[32mRemoving images...\033[m")
            try:
                remove_img_augmentation()
                print("\033[32mImages removed!\033[m")

            except Exception as e:
                print(f"\033[31mError removing imgs: {e}\033[m")

        elif choice == 3:
            print("\033[32mConverting images...\033[m")
            try:
                png_converter()
                print("\033[32mImages comverted!\033[m")

            except Exception as e:
                print(f"\033[31mError converting imgs: {e}\033[m")
        else:
            return 0
    except:
        print("\033[31mStoping...\033[m")
main()