# MNIST Digit Recognition Model

This project implements a deep learning model for digit recognition using the MNIST dataset. The model is built with TensorFlow and Keras, and it uses a Convolutional Neural Network (CNN) architecture to classify handwritten digits from 0 to 9. The model is trained on the MNIST dataset and evaluated on a test set to predict the accuracy of the classification.


## Project Overview

The goal of this project is to classify images of handwritten digits using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits. After training, the model's performance is evaluated on the test set, and the accuracy is reported.

## Setup Instructions

1. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/csm34/MNIST-Digits.git
    cd MNIST-Digits
    ```

2. **Install dependencies**:

    Ensure that you have Python 3.x and the following libraries installed:

    - TensorFlow
    - Matplotlib
    - Numpy
    - Pandas

    You can install the dependencies using `pip`:

    ```bash
    pip install tensorflow matplotlib numpy pandas
    ```

## Dataset

This project uses the **MNIST** dataset, which consists of 28x28 grayscale images of handwritten digits from 0 to 9. The dataset is available in TensorFlow and Keras and is loaded using:


## Model Architecture
The model is a simple Convolutional Neural Network (CNN) built using the Keras API:

- Conv2D Layer: 28 filters of size 3x3 with ReLU activation.
- MaxPooling2D Layer: Pooling with a 2x2 window.
- Flatten Layer: Flattens the output of the convolutional layers to a 1D vector.
- Dense Layer: Fully connected layer with 128 units and ReLU activation.
- Output Layer: Fully connected output layer with 10 units (one for each digit) and softmax activation.

## Training the Model
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. 
It is trained for 10 epochs with a batch size of 64. The training process includes both training and validation on the test set.


## License
This project is licensed under the MIT License.
