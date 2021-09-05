import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
import base64

st.title(' ')
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
       data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
        <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
        ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('image.png')


def classifier(img, file):
    np.set_printoptions(suppress=True)
    model = keras.models.load_model(file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


st.title("Waste Classifier")
uploaded_file = st.file_uploader(" ", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded IMage.', use_column_width=True)
    st.write("Classifying Image")
    label = classifier(image, 'model.h5')
    cardboard= (label[0][0])
    glass= (label[0][1])
    metal= (label[0][2])
    paper= (label[0][3])
    plastic= (label[0][4])
    trash= (label[0][5])
    if cardboard >= 0.6:
        st.title("It is cardboard")
    elif glass >= 0.6:
        st.title("It is glass")
    elif metal >= 0.6:
        st.title(" It is metal")
    elif paper >= 0.6:
        st.title("It is paper")
    elif plastic >= 0.6:
        st.title("It is plastic")
    elif trash >= 0.6:
        st.title("It is trash")
