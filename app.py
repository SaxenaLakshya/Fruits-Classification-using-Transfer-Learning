import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the model
model = tf.keras.models.load_model("Fruit_Classifier_1o.h5")

def predict_class(image):
    classes = {0: 'Apple', 1: 'Banana', 2: 'Grape', 3: 'Mango', 4: 'Strawberry'}
    img = load_img(image, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return classes[predicted_class_index] if confidence > 0.5 else 'Unknown'


if __name__ == "__main__":
    st.title("Welcome to Fruit Classification Model!")
    st.write("Remember: This model is 65% Accurate so, the predicted output might be wrong.")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        output = predict_class(uploaded_file)
        st.write(f"Model has predicted the uploaded image as {output}.")