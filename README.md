# Introduction
This project is a Fashion Recommender System built using classic machine learning and deep learning techniques. It suggests fashion items to users based on the images that they upload.
I developed a fashion recommender system utilizing ResNet, leveraging the power of transfer learning. The model extracts 2048-dimensional feature vectors from input images. I then used Streamlit to create a user-friendly interface where users can upload an image. This image is processed by ResNet to extract features, and recommendations are generated using the K-Nearest Neighbors algorithm

# Features
Personalized Recommendations: Offers tailored fashion recommendations.
User Interaction: Allows users to input their preferences.
Fashion Dataset: Utilizes a curated fashion dataset.
Real-time Suggestions: Provides instant recommendations as users interact with the system.

# Demo
[Check out the fashion recommender system](https://huggingface.co/spaces/Manasa1/Fashion_Recommender_System)

![Screenshot (11)](https://github.com/user-attachments/assets/9f4898bb-9c56-450f-a7a3-9840c145cfd0)

![Screenshot (12)](https://github.com/user-attachments/assets/91ff4378-286d-4401-8190-a509098bc21b)

# Dataset
Used Fashion Product Images Dataset by Param Aggarwal
[Dataset Link](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

# How does this work
![fashion drawio](https://github.com/user-attachments/assets/c3ab142d-79d1-44cc-bcdd-900eec1f3dbb)

# Steps included

## 1. Getting the dataset
This dataset is taken from Kaggle. This image dataset has 44 K images. Here is a sample of images from the Fashion Product Images Dataset

![Screenshot (10)](https://github.com/user-attachments/assets/eeb9ee08-2ca8-4ee5-b5ad-19f9cd20bac2)

## Resnet
Once the data is preprocessed, feed it to the ResNet model

## Extracting Features
Extracting Features using ResNet

## Recommendation Generation

To generate recommendations, our proposed approach utilizes the Scikit-learn Nearest Neighbors algorithm. This allows us to find the nearest neighbors for a given input image. The similarity measure used in this project is Cosine Similarity. The top 5 recommendations are then extracted from the database, and their images are displayed.

## Experiment and Results

The concept of Transfer Learning is employed to address the challenges posed by the small size of the fashion dataset. We pre-train the classification models on the DeepFashion dataset, which consists of 44,441 garment images. The networks are trained and validated on the selected dataset. The training results demonstrate high model accuracy, with low error and loss, and a strong F-score

# Installation
Use pip to install the requirements.
''' 
pip install -r requirements.txt
'''











