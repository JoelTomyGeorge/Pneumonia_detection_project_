import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model from the pickle file
# model_path = "models/pneumonia_model.pkl"
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)
    
# from tensorflow.keras.models import load_model
# # Load the saved model
# model = load_model('pneumonia_detection_model.h5')

from tensorflow.keras.models import load_model
# Load the saved model
model = load_model('better_models/pneumonia_detection_cnn_model.h5')



# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Streamlit interface
st.title("Pneumonia Detection")
st.write("Upload an image of a chest X-ray to predict if it shows pneumonia.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file)
    processed_image = preprocess_image(img)

    # Make prediction
    prediction = model.predict(processed_image)

    # Display the prediction result
    if prediction[0][0] > 0.5:
        st.success("Predicted: Pneumonia")
    else:
        st.success("Predicted: Normal")