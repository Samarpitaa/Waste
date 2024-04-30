import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the class labels
class_labels = ['Keyboards','Mobile', 'Mouses', 'TV', 'camera','laptop', 'microwave', 'smartwatch']

# Function to preprocess image for prediction
def preprocess_image(img):
    img = img.resize((384, 384))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to classify image using the loaded model
def classify_image(model, img):
    img = preprocess_image(img)
    pred = model.predict(img)[0] 
    max_prob_idx = np.argmax(pred)

    max_prob_class = class_labels[max_prob_idx]
    max_prob = pred[max_prob_idx]
    return max_prob_class, max_prob


model = load_model('model.h5') 

# Streamlit app
st.title('Image Classification App')

# Function to upload image
def upload_image():
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        return img
    return None

# Main app logic
uploaded_img = upload_image()
if uploaded_img is not None:
    # Display uploaded image
    st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)
    # Classify image
    pred_class, pred_prob = classify_image(model, uploaded_img)
    # Display classification results
    st.write('Prediction:')
    st.write(f'Class: {pred_class}, Probability: {pred_prob:.2f}')







