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

ref = "Find the whole code [here](https://github.com/ovh/ai-training-examples/tree/main/apps/gradio/sketch-recognition)."

img_size = 28
labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

model = tf.keras.models.load_model("number_recognition_model_colab.keras")


def predict(img):
    try:
        # Convert the input image to a NumPy array if needed
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Ensure the image has a single channel (grayscale)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Print the type and shape of the image
        print(type(img), img.shape)

        # Resize the image
        img = cv2.resize(img, (img_size, img_size))
        img = img.reshape(1, img_size, img_size, 1)
        preds = model.predict(img)[0]
        return {label: float(pred) for label, pred in zip(labels, preds)}
    except Exception as e:
        # Print the exception to the console
        print(f"Error during prediction: {e}")
        return {"Error": str(e)}


label = gr.Label(num_top_classes=3)

interface = gr.Interface(fn=predict, inputs="sketchpad", outputs=label, title=title, description=head, article=ref)
interface.launch()