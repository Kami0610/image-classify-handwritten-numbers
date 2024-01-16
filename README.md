# Computer Vision: Image Classification of Handwritten Numbers

I created this project to better learn the process of training, testing and using image classification models. This 
project uses the MNIST dataset of handwritten numbers between 0 and 9 to train a computer vision model. The code has 
been split into multiple files, including: model training, model accuracy testing and user image testing.

## Project Files
- **constants.py**
  - Contains the constant values that are used throughout the rest of the project.
- **model_test.py**
  - Tests the accuracy of a trained model. It uses the MNIST testing dataset of 10,000 images.
- **model_train.py**
  - Trains a computer vision model, using the MNIST training dataset of 60,000 images.
- **neural_network.py**
  - Convolutional neural network class.
- **user_image_testing.py**
  - Gets the user provided images and displays the prediction percentages.
- **computer_vision_notebook.ipynb**
  - Notebook of the training and testing the computer vision model.

## Test Your Own Images
To test your own handwritten images, place them into the `images` folder. When running the user_image_testing.py, it 
will first walk through the `images` folder and get all image files. The program then proceeds use the trained model to 
make a prediction. By default, the program displays only the highest 3 prediction percentages.

## User Image Testing Output Examples
```text
The image: ../images/image_0.png
Prediction number: 0 --> 23.19%
Prediction number: 6 --> 8.54%
Prediction number: 1 --> 8.53%
========================================
The image: ../images/image_1.png
Prediction number: 1 --> 23.20%
Prediction number: 2 --> 8.53%
Prediction number: 0 --> 8.53%
========================================
The image: ../images/Image_2.png
Prediction number: 2 --> 23.20%
Prediction number: 1 --> 8.53%
Prediction number: 0 --> 8.53%
========================================
The image: ../images/Image_3.png
Prediction number: 3 --> 23.20%
Prediction number: 1 --> 8.53%
Prediction number: 0 --> 8.53%
========================================
The image: ../images/Image_4.png
Prediction number: 4 --> 23.20%
Prediction number: 1 --> 8.53%
Prediction number: 0 --> 8.53%
========================================
The image: ../images/Image_5.png
Prediction number: 5 --> 23.08%
Prediction number: 9 --> 8.59%
Prediction number: 3 --> 8.54%
========================================
```