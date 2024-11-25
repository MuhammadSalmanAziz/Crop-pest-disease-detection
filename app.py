import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
import tensorflow as tf

# Load the trained VGG19 model
@st.cache(allow_output_mutation=True)
def load_vgg19_model():
    return load_model('vgg19_trained_model.h5')  # Replace with the path to your trained model file

model = load_vgg19_model()

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match input size of VGG19
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image according to VGG19 requirements
    return img_array

# Make predictions
def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(preds, axis=1)[0]
    return predicted_class_index

Class_names = ['Cassava Healthy', 'Cashew Anthracnose', 'Tomato Healthy', 'Cashew Leaf Miner', 'Cashew Gumosis', 'Cassava Brown Spot', 'Maize Grasshoper', 'Maize Healthy', 'Maize Fall Armyworm', 'Tomato Verticulium Wilt', 'Cassava Mosaic', 'Maize Leaf Beetle', 'Cassava Bacterial Blight', 'Cashew Red Rust', 'Tomato Septoria Leaf Spot', 'Maize Leaf Spot', 'Maize Streak Virus', 'Tomato Leaf Curl', 'Tomato Leaf Blight', 'Cassava Green Mite', 'Cashew Healthy', 'Maize Leaf Blight']

# Streamlit app
def main():
    st.title("Crop Disease Detection")
    st.write("Upload an image of a crop to detect the disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Disease'):
            st.write("Processing...")
            predicted_class_index = predict(image)
            predicted_class_name = Class_names[predicted_class_index]
            st.write("Predicted class:", predicted_class_name)

if __name__ == '__main__':
    main()
