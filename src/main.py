import CNNModel
import Predict
import os
import logging

logging.basicConfig(level=logging.INFO)

path = "../Images"
db_path = os.path.join(os.getcwd(),"../DataSet/")


def list_files():
    files = []
    i=0

    if not os.path.exists(path):
        logging.warning(f"Path does not exists: {path}")
        return None

    for file in os.listdir(path):
        print(f"{i}: {file}")
        file_path = os.path.join(path,file)
        files.append(file_path)
        i+=1

    return files

def remove_trash():
    global db_path
    boolean = False
    db_path = os.path.join(db_path,"Train")

    if not os.path.exists(db_path):
        logging.warning(f"\033[31mPath does not exists: {db_path}\033[m")
        return

    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path,folder)
        if os.path.isdir(folder_path): # check if it is a directory
            for img in os.listdir(folder_path):
                if not img.endswith(".png"):
                    img_path = os.path.join(folder_path,img)
                    try:
                        boolean = True
                        os.remove(img_path)

                    except Exception as e:
                        logging.error(f"\033[31mError removing trash {img}: {e}\033[m")

    if boolean == True:
        print(f"\033[32mTrash removed!\033[m")



def main():
    try:
        choice = 0
        while True:
            choice = input("1: Train model\n2: Check image\nInput: ")
            if choice.isdigit():
                choice = int(choice)
                if choice == 27:
                    break
                if choice > 2 or choice < 1:
                    print("\033[31mInvalid!\033[m")
                else:
                    break
            else:
                print("\033[31mInvalid!\033[m") 
        if choice == 1:
            remove_trash()
            print("\033[32mTraining model...\033[m")
            try:
                model, X_t1, X_t2, y_t1, y_t2 = CNNModel.model_builder()
                CNNModel.model_compilation(model, X_t1, X_t2, y_t1, y_t2)
                CNNModel.accuracy_test(model)
                print("\032[32mModel trained and tested successfully.\033[m")
                
            except Exception as e:
                logging.error(f"\033[31mError during model training: {e}\033[m")

        elif choice == 2:
            traf_sign = None
            img_opt = 0

            model = Predict.my_load_model()
            if not model:
                return 
            
            files = list_files()
            if files is None:
                return 
            
            while True:
                img_opt = input("Choose one image: ")
                if img_opt.isdigit():
                    img_opt = int(img_opt)
                    if (img_opt < 0 or img_opt > len(files)-1) != False:
                        print("\033[31mInvalid!\033[m")
                    else: 
                        break
                else:
                    print("\033[31mInvalid!\033[m")

            img_path = files[img_opt]

            traf_sign = Predict.classify_img(model,img_path)

            print(traf_sign)
        
        elif choice == 27:
            model = Predict.my_load_model()
            for img in os.listdir(path):
                if "DS_Store" not in img:
                    file_path = os.path.join(path,img)
                    traf_sign = Predict.classify_img(model,file_path)
                    print(f"\033[33m{img} -> {traf_sign}\033[m")

    except Exception as e:
        logging.error(f"\033[31mError in main: {e}\033[m")


if __name__ == "__main__":
    main()
