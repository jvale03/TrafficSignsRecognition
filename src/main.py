import CNNModel
import Predict
import os

path = "../Images"

def list_files():
    files = []
    i=0
    for file in os.listdir(path):
        print(f"{i}: {file}")
        file_path = os.path.join(path,file)
        files.append(file_path)
        i+=1

    return files

def main():
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
        print("\033[32mTraining model!\033[m")
        try:
            CNNModel.model_builder()
            CNNModel.model_compilation()
            CNNModel.accuracy_test()
            
        except:
            print("\033[31mError creating model!\033[m")

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
                print(f"{img} -> {traf_sign}")




main()
