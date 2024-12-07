import cv2
import gradio as gr
import tensorflow as tf
import numpy as np

title = "Welcome on your first sketch recognition app!"

head = (
    "<center>"
    "The robot was trained to classify numbers (from 0 to 9). To test it, write your number in the space provided."
    "</center>"
)


img_size = 28
labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

model = tf.keras.models.load_model("number_recognition_model_colab.keras")

def predict(img):
    try:
        # Convert the input image to a NumPy array if needed
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Print shape and type of the input image
        print(f"Initial image type: {type(img)}, shape: {img.shape}")

        # Ensure the image is in grayscale and has a single channel
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Print the shape of the grayscale image
        print(f"Grayscale image shape: {img.shape}")

        # Resize the image
        img = cv2.resize(img, (img_size, img_size))

        # Normalize the image
        img = img.astype('float32') / 255.0
        img = img.reshape(1, img_size, img_size, 1)

        # Print the shape after resizing and normalizing
        print(f"Processed image shape: {img.shape}")

        preds = model.predict(img)[0]

        # Print the predictions
        print("Predictions:", preds)

        return {label: float(pred) for label, pred in zip(labels, preds)}
    except Exception as e:
        # Print the exception to the console
        print(f"Error during prediction: {e}")
        return {"Error": str(e)}

label = gr.Label(num_top_classes=3)

interface = gr.Interface(fn=predict, inputs="sketchpad", outputs=label, title=title, description=head)
interface.launch(debug=True)