from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
import os
from datetime import datetime
from PIL import Image
import logging
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)

db_path = os.path.join(os.getcwd(), "../DataSet/")

def imgs_augmentation_train():
    global db_path
    db_path = os.path.join(db_path, "Train")
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):  # Check if it is a directory
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                try:
                    gen_images(img_path, folder_path)
                except (OSError, Exception) as e:
                    logging.error(f"\033[31mError generating images for {img_path}: {e}\033[m")

def remove_img_augmentation():
    global db_path
    db_path = os.path.join(db_path, "Train")
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):  # Check if it is a directory
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                if "image" in img:
                    try:
                        os.remove(img_path)
                    except (OSError, Exception) as e:
                        logging.error(f"\033[31mError removing image {img_path}: {e}\033[m")

def png_converter():
    global db_path
    db_path = os.path.join(db_path, "Train")
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):  # Check if it is a directory
            for image in os.listdir(folder_path):
                if image.endswith((".jpg", ".jpeg")):
                    img_path = os.path.join(folder_path, image)
                    try:
                        with Image.open(img_path) as img:
                            nome_ficheiro = os.path.splitext(image)[0]
                            img.save(os.path.join(folder_path, nome_ficheiro + ".png"))
                        os.remove(img_path)
                    except (OSError, Exception) as e:
                        logging.error(f"\033[31mError converting image {img_path}: {e}\033[m")

def add_noise(img_array):
    noise_factor = 0.05
    noise = np.random.randn(*img_array.shape) * noise_factor
    img_array = img_array + noise
    img_array = np.clip(img_array, 0., 255.)
    return img_array

def gen_images(img_path, dir):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=[0.65,1.15],
        channel_shift_range=10.0,
        brightness_range=(0.7, 1.3))

    img = load_img(img_path)
    img_array = img_to_array(img)
    img_array = add_noise(img_array)
    img_array = img_array.reshape((1,) + img_array.shape)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    i=0
    for batch in datagen.flow(img_array, batch_size=1,
                              save_to_dir=dir,
                              save_prefix=f'image{timestamp}{i}', save_format='png'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        i+=1
        if i >= 3:
            break

def main():
    try:
        choice = 0
        while True:
            choice = input("1: Image augmentation\n2: Remove created images\n3: Convert to png\n4: Cancel\nInput: ")
            if choice.isdigit():
                choice = int(choice)
                if choice > 4 or choice < 1:
                    print("\033[31mInvalid choice!\033[m")
                else:
                    break
            else:
                print("\033[31mInvalid input!\033[m")

        if choice == 1:
            print("\033[32mGenerating images...")
            imgs_augmentation_train()
            print("\033[32mImages generated!\033[m")

        elif choice == 2:
            print("\033[32mRemoving images...\033[m")
            remove_img_augmentation()
            print("\033[32mImages removed!\033[m")

        elif choice == 3:
            print("\033[32mConverting images...\033[m")
            png_converter()
            print("\033[32mImages converted!\033[m")

        elif choice == 4:
            return None
        
    except Exception as e:
        logging.error(f"\033[31mError in main: {e}\033[m")

if __name__ == "__main__":
    main()
