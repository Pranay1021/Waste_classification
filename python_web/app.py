import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import requests
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# model = pickle.load(open('../final_model.pkl', 'rb'))
model =  tf.keras.models.load_model('../final_model.h5')
output_class = ["Hazardous","NR","Organic","Recycle"]
def waste_prediction(new_image):
    test_image = Image.open(new_image)
    test_image = test_image.resize((256, 256))
    
    #test_image = image.img_to_array(test_image) / 255
    test_image = np.array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)
    return predicted_value, predicted_accuracy

st.title('Garbage Classification')
st.write('This is a simple image classification web app to classify the type of garbage')

file = st.file_uploader("Please upload an image file", type=["jpg", "g"])
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    label, accuracy = waste_prediction(file)

    st.write('The garbage is classified as ', label, ' with ', accuracy, '% accuracy')

