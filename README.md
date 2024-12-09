# MNIST Digit Recognition Project

This repository contains code for training, evaluating, and deploying a convolutional neural network (CNN) to recognize handwritten digits from the MNIST dataset. The project includes the following features:

- Training a CNN model using TensorFlow and Keras.
- Saving the trained model for future use.
- Building an interactive web application using Gradio for digit recognition.

Hugging face space: https://huggingface.co/spaces/cisemh/Mnist-Digits

---

## Dataset

The model is trained and tested on the MNIST dataset, a standard dataset in machine learning that consists of grayscale images of handwritten digits (0-9). Each image is 28x28 pixels.

- **Training Set:** 60,000 images
- **Test Set:** 10,000 images

## File Structure

### 1. `mnist-digits.ipynb`
This Jupyter Notebook contains:
- Loading and preprocessing the MNIST dataset.
- Defining and training a CNN model.
- Evaluating the model’s performance.
- Saving the model in `.keras` and `.pkl` formats.

### 2. `app.py`
This Python script uses Gradio to create a web-based interface where users can draw digits and get predictions from the trained model.

- **Dependencies:**
  - Gradio v3.50.2
  - TensorFlow
  - NumPy
  - Matplotlib

- **Features:**
  - Load the trained model.
  - Predict the digit drawn by the user on a canvas.

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/csm34/MNIST-Digits.git
   cd MNIST-Digits
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook to train and save the model (if not already done):
   ```bash
   jupyter notebook mnist-digits.ipynb
   ```

4. Launch the Gradio app:
   ```bash
   python app.py
   ```

### Interacting with the Application

- The Gradio interface will open in your web browser.
- Use the canvas to draw a digit (0-9).
- The app will predict the digit and display the top 3 predictions with their probabilities.

## Model Architecture

The CNN model has the following layers:
1. Conv2D: 28 filters, kernel size (3, 3), ReLU activation
2. MaxPooling2D: Pool size (2, 2)
3. Flatten: Converts 2D features into a 1D vector
4. Dense: 128 units, ReLU activation
5. Dense: 10 units (output layer), softmax activation

### Training Parameters
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 10
- Batch Size: 64

### Results

- *Final Test Accuracy: 98.27%
- Test Error Rate: 1.31%

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- Gradio v3.50.2

You can install the dependencies using the command:
```bash
pip install tensorflow numpy matplotlib gradio==3.50.2
```



## License
This project is licensed under the MIT License.
