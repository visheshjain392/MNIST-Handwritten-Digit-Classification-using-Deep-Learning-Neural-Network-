🖋 MNIST Handwritten Digit Classification
A deep learning project for classifying handwritten digits (0–9) from the MNIST dataset using TensorFlow/Keras. Implemented and tested in Google Colab, with visualizations, preprocessing, model training, and performance evaluation.

📌 Project Overview
The MNIST dataset contains 70,000 grayscale images of handwritten digits. This project builds a Neural Network model to accurately classify these digits, leveraging Python libraries for EDA, visualization, and deep learning.

🛠 Tech Stack
Programming Language: Python

Libraries: NumPy, Matplotlib, Seaborn, OpenCV, Pillow

Deep Learning: TensorFlow, Keras

Environment: Google Colab

📂 Features
Load and preprocess MNIST dataset

Data visualization using Matplotlib & Seaborn

Neural Network model creation with TensorFlow/Keras

Model training and evaluation

Confusion matrix for performance insights

📊 Workflow
Import Libraries

python
Copy
Edit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix
Load & Explore Data

Preprocess Images

Build Neural Network Model

Train & Validate

Evaluate Model Performance

📈 Results
High accuracy achieved on test set

Confusion matrix visualized to analyze misclassifications

Clear improvement over baseline models

🚀 How to Run in Google Colab
Open Google Colab

Upload the .ipynb notebook

Run each cell sequentially

View results and visualizations directly in the notebook

📌 Future Improvements
Experiment with CNN models for better accuracy

Apply data augmentation to improve generalization

Deploy model as a web app for real-time digit prediction

