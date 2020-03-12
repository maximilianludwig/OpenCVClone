import os
import Constants
import cv2 as cv
import numpy as np

from flask import Flask, request, render_template
from werkzeug import secure_filename

from tensorflow.keras.models import load_model
from PIL import Image
from lib import count_files, image_to_pixel_array, load_prediction_dict

APP = Flask(__name__, template_folder=Constants.TEMPLATE_FOLDER, static_url_path="", static_folder=Constants.UPLOAD_FOLDER)
MODEL = None
PREDICTION_DICT = None

def init():
    global MODEL
    global PREDICTION_DICT
    MODEL = load_model(os.path.join(Constants.MODEL_FOLDER, Constants.BEST_MODEL))
    PREDICTION_DICT = load_prediction_dict(Constants.BEST_MODEL)
    if not os.path.isdir(os.path.join(Constants.UPLOAD_FOLDER, Constants.UPLOAD_PROCESSED_FOLDER)):
        os.makedirs(os.path.join(Constants.UPLOAD_FOLDER, Constants.UPLOAD_PROCESSED_FOLDER))

def image_preprocessing(file_name):
    image = cv.imread(os.path.join(Constants.UPLOAD_FOLDER, file_name))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if gray.shape[:2] == (48, 48):
        cv.imwrite(os.path.join(Constants.UPLOAD_FOLDER, Constants.UPLOAD_PROCESSED_FOLDER, file_name), gray)

    face_first = cv.CascadeClassifier(Constants.FACE_DETECTION_FIRST).detectMultiScale(gray, minNeighbors=10, minSize=(20, 20), flags=cv.CASCADE_SCALE_IMAGE)
    face_two =  cv.CascadeClassifier(Constants.FACE_DETECTION_SECOND).detectMultiScale(gray, minNeighbors=10, minSize=(20, 20), flags=cv.CASCADE_SCALE_IMAGE)
    face_three = cv.CascadeClassifier(Constants.FACE_DETECTION_THIRD).detectMultiScale(gray, minNeighbors=10, minSize=(20, 20), flags=cv.CASCADE_SCALE_IMAGE)
    face_four = cv.CascadeClassifier(Constants.FACE_DETECTION_FOURTH).detectMultiScale(gray, minNeighbors=10, minSize=(20, 20), flags=cv.CASCADE_SCALE_IMAGE)

    if len(face_first) == 1:
        facefeatures = face_first
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        return False

    (x, y, w, h) =  facefeatures[0]
    gray_face = gray[y:y+h, x:x+w]
    gray_face_resized = cv.resize(gray_face, (Constants.IMG_SIZE, Constants.IMG_SIZE))
    cv.imwrite(os.path.join(Constants.UPLOAD_FOLDER, Constants.UPLOAD_PROCESSED_FOLDER, file_name), gray_face_resized)
    return True

@APP.route("/")
def home():
    return render_template("image-selector.html")

@APP.route('/image-selector', methods=['GET', 'POST'])
def form_post():
    return render_template("image-selector.html")

@APP.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        file_name = str(count_files(Constants.UPLOAD_FOLDER) + 1) + ".png"
        file.save(os.path.join(Constants.UPLOAD_FOLDER, file_name))
        if image_preprocessing(file_name):
            arr = image_to_pixel_array(os.path.join(Constants.UPLOAD_FOLDER, Constants.UPLOAD_PROCESSED_FOLDER, file_name))
            arr = np.expand_dims(arr, axis=0)
            prediction = MODEL.predict(arr)
            emotion = PREDICTION_DICT[np.argmax(prediction[0])]
            prediction_float = list(map(lambda x: '{0:.10f}'.format(float(x)), prediction[0]))
            '''printable_prediction = "[ "
            for i in range(len(prediction[0])):
            printable_prediction += Constants.NUMBER_TO_EMOTION[i] + ": " + '{0:.4f}'.format(float(prediction[0][i])) + ", "
            printable_prediction = printable_prediction[:-2] + " ]"'''
            return render_template("prediction.html", predicted_values=prediction_float, PREDICTION_DICT=PREDICTION_DICT, file_name=file_name)
        else:
            return render_template("no-face.html")

init()
APP.run(debug=False, host=Constants.HOST, port=Constants.PORT)