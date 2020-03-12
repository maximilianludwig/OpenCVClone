import pandas
import os

# DATA
RAW_DATA = pandas.read_csv("Code\\fer2013.csv")
COLUMN_EMOTION = "emotion"
COLUMN_USAGE = "Usage"
COLUMN_PIXELS = "pixels"

# MODEL
BEST_MODEL = "Best.model"
MODEL_FOLDER = "models"
MODEL_METADATA_FILE = "METADATA"
EPOCHS = 7
BATCH_SIZE = 64
IMG_SIZE = 48

# EMOTIONS
EMOTIONS = [ "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral" ]
EMOTION_TO_NUMBER = { "Angry":0, "Disgust":1, "Fear":2, "Happy":3, "Sad":4, "Surprise":5, "Neutral":6 }
NUMBER_TO_EMOTION = { 0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral" }


# WEBSERVER
PORT = 8080
HOST = '0.0.0.0'
UPLOAD_FOLDER = "uploads"
UPLOAD_PROCESSED_FOLDER = "processed"
TEMPLATE_FOLDER = "html"
FACE_DETECTION_BASE_DIR = "haarcascades"
FACE_DETECTION_FIRST = os.path.join(FACE_DETECTION_BASE_DIR, "haarcascade_frontalface_default.xml")
FACE_DETECTION_SECOND = os.path.join(FACE_DETECTION_BASE_DIR, "haarcascade_frontalface_alt2.xml")
FACE_DETECTION_THIRD = os.path.join(FACE_DETECTION_BASE_DIR, "haarcascade_frontalface_alt.xml")
FACE_DETECTION_FOURTH = os.path.join(FACE_DETECTION_BASE_DIR, "haarcascade_frontalface_alt_tree.xml")

# OTHER
PICTURE_FOLDER = "picture"




