import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the model (make sure this file is uploaded in your Space)
model = tf.keras.models.load_model("Adam_run22_model.h5")

class_names = ['Drowsy', 'Non Drowsy']

def classify_image(img):
    img = image.img_to_array(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    return f"Class: {predicted_class}"

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Driver Image"),
    outputs="text",
    title="Driver Drowsiness Detector",
    description="Upload an image to check if the driver is drowsy or not."
).launch()
