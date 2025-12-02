import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
#from huggingface_hub import from_pretrained_keras
# Load the model (make sure this file is uploaded in your Space)
from huggingface_hub import hf_hub_download

# Download the model file from your Hugging Face repo
model_path = hf_hub_download(repo_id="Mielle85/MobileDDD", filename="Adam_run22_model.h5")

# Load it with Keras
model = tf.keras.models.load_model(model_path)

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
