from functions import *
import joblib
import numpy as np

rf_classifier = joblib.load('random_forest_model.pkl')

img = preprocess_image('./Test Images/N2.jpg', 144, 144)
img = img.reshape(1, -1)
prediction = rf_classifier.predict(img)
prediction[0]