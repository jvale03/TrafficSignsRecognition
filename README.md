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
In our `train` folder, are available around 43 subfolders and each one represents a different class, in this case, different traffic sign. With this, we can see that our data shape is `(X, Y, Z, 3)`, `X` represents the number of images, `Y*Z` represents the image sizes in pixels and `3` represents the RGB values.

## Images augmentation
Although the model's accuracy is 97%, without this data processing, the model does not match many random traffic signs so I realized that a training dataset with 30k images wasn't enough. 

With image augmentation we can come up with new transformed images from our original dataset, changing `rotation`, `shifts`, `flips`, `brightness`, etc. This way, we can have a dataset with many more images for model training.

This processing needs to be executed with another program before training the model, running `python Augmentation.py`.

## CNN model
A CNN (Convolutional Neural Network) is a machine learning model mainly used for tasks involving visual data, such as **image** and video **recognition**. It's a variation of neural networks that uses convolutions instead of simple matrix multiplications, making it ideal for processing grid-like data structures, like image pixels. [Lean More](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/)

## Prediction

## Refs:
1. DataSet: [link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
2. CNN: [link](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/)
3. Data Augmentation: [link](https://www.geeksforgeeks.org/python-data-augmentation/) and [link](https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/)
4. Tutorial WebSite: [link](https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/)

