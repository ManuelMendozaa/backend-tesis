from flask import Flask

# Setting up application
app = Flask(__name__)
app.config.from_object("config.DevelopmentConfig")

# Loading ML models
import cv2
from app.models.GoogleNet import googlenet
from app.models.FaceNet import InceptionResnetV1
FED_model, c_model = googlenet()
FD_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facenet = InceptionResnetV1().eval()

# Getting routes
from app import views