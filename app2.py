import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
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

# Sidebar for additional options
st.sidebar.title("Options")
st.sidebar.header("About the Model")
st.sidebar.write("This model is trained to detect pneumonia from chest X-ray images.")
st.sidebar.write("Accuracy: 92%")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        img = image.load_img(uploaded_file)
        processed_image = preprocess_image(img)

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_container_width=True, width =250)

        # Make prediction
        with st.spinner("Making prediction..."):
            prediction = model.predict(processed_image)

        # Display the prediction result
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        st.write(f"Confidence: {confidence:.2f}")

        if prediction[0][0] > 0.75:
            st.success("Predicted: Pneumonia")
        else:
            st.success("Predicted: Normal")

        # Download button for prediction result
        st.download_button("Download Result", "Predicted: Pneumonia" if prediction[0][0] > 0.5 else "Predicted: Normal")

    except Exception as e:
        st.error(f"Error: {e}")

# Reset button
if st.button("Reset"):
    st.experimental_rerun()
