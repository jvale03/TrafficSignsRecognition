# TrafficSignsRecognition

Machine learning project that predicts a **Traffic Sign** in an image.  

## Run project

Run the main file `main.py` and make sure all paths are right. You can use the model I provided in this repository, or create your own model with your dataset, being carefully with the dataset exploration fase. 


## Project tree

```
- /TrafficSignsRecognition
    - /src
        - ...
    - /DataSet
        - /Meta
        - /Test
        - /Train
        - ...
    - /Images
        - ...
    - /Model
        - ...

```

- `/src` -> directory with our code
- `/DataSet` -> directory with all necessary data to train and test our model.
- `/Images` -> directory with images to test our model.
- `/Model` -> directory with models to predict our images.


## Dataset exploration
In our `train` folder, are available around 43 subfolders and each one represents a different class, in this case, different traffic sign. With this, we can see that our data shape is `(39209, 30, 30, 3)`, `39209` represents the number of images, `30*30` represents the image sizes in pixels and `3` represents the RGB values.

## CNN model



## Refs:
1. DataSet: [link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
2. CNN: [link](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/)
3. Tutorial WebSite: [link](https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/)

