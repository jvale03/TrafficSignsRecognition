import CNNModel
import Predict
import os

path = "../Images"
db_path = os.path.join(os.getcwd(),"../DataSet/")


def list_files():
    files = []
    i=0
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
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path,folder)
        if os.path.isdir(folder_path): # check if it is a directory
            for img in os.listdir(folder_path):
                if not img.endswith(".png"):
                    boolean = True
                    img_path = os.path.join(folder_path,img)
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"\033[31mError removing trash: {e}\033[m")
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
                CNNModel.model_builder()
                CNNModel.model_compilation()
                CNNModel.accuracy_test()
                
            except Exception as e:
                print(f"\033[31mError creating model: {e}\033[m")

        elif choice == 2:
            traf_sign = None
            img_opt = 0

            files = list_files()
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

            model = Predict.my_load_model()
            traf_sign = Predict.classify_img(model,img_path)

            print(traf_sign)
        
        elif choice == 27:
            model = Predict.my_load_model()
            for img in os.listdir(path):
                if "DS_Store" not in img:
                    file_path = os.path.join(path,img)
                    traf_sign = Predict.classify_img(model,file_path)
                    print(f"\033[33m{img} -> {traf_sign}\033[m")

    except:
        print("\033[31mStoping...\033[m")


main()
