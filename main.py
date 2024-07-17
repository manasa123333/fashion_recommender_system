import streamlit as st
import os
from PIL import Image
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
try:
    with open(filename, "rb") as f:
        return f.read()
except FileNotFoundError:
    raise MediaFileStorageError(f"File '{filename}' not found.")
except PermissionError:
    raise MediaFileStorageError(f"Permission denied for '{filename}'.")
except Exception as ex:
    raise MediaFileStorageError(f"Error opening '{filename}': {ex}")

feature_list = np.array(pickle.load(open('embeddings2.pkl', 'rb')))
filenames = pickle.load(open('filenames2.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title("Fashion Recommender System")


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(expanded_image_array)
    result = model.predict(processed_image).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


uploaded_file = st.file_uploader("choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = extract_features(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        indices = recommend(features,feature_list)
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error has occured while uploading file")
