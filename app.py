import numpy as np
import tensorflow

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights="imagenet", include_top=False,input_shape=(224,224,3))
model.trainable=False

model1 = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def  extract_features(img_path,model):
    img=image.load_img(img_path,target_size = (224,224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array,axis=0)
    processed_image = preprocess_input(expanded_image_array)
    result = model.predict(processed_image).flatten()
    normalized_result=result/norm(result)
    return  normalized_result

filenames =[]

for file in os.listdir('set0'):
    filenames.append(os.path.join('set0',file))

for file in os.listdir('set1'):
    filenames.append(os.path.join('set1',file))

for file in os.listdir('set2'):
    filenames.append(os.path.join('set2',file))

for file in os.listdir('set3'):
    filenames.append(os.path.join('set3',file))

for file in os.listdir('set4'):
    filenames.append(os.path.join('set4',file))

feature_list = []

for i in tqdm(filenames):
    feature_list.append(extract_features(i,model1))

print(np.array(feature_list).shape)

import pickle
pickle.dump(feature_list,open('embeddings2.pkl','wb'))
pickle.dump(filenames,open('filenames2.pkl','wb'))