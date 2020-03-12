import glob, random, time
import numpy as np
import cv2 as cv

emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def get_training_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        emotion_number = emotions.index(emotion)
        training = glob.glob("dataset/%s/*" % emotion)
        for item in training:
            training_data.append(cv.imread(item))
            training_labels.append(emotion_number)
    return training_data, training_labels

def create_fishface():
    return cv.face_FisherFaceRecognizer.create()

def train_classifier(fishface, data, labels):
    fishface.train(data, np.asarray(labels))
    return fishface

def test_classifier(fishface):
    cam = cv.VideoCapture(0)
    cv.namedWindow("Classifier Test")
    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = gray[y:y + h, x:x + w]
            try:
                gray = cv.resize(gray, (350, 350))
                prediction, conf = fishface.predict(gray)
                cv.putText(img, emotions[prediction], (x, h), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv.LINE_AA)
            except:
                pass
        cv.imshow("Classifier Test", img)
        if not ret:
            break

        k = cv.waitKey(1)
        if k%256 == 27: # ESC pressed, closing the window
            cam.release()
            cv.destroyAllWindows()
            break

def save_classifier():
    fishface.write("second_fisher_face_classifier.xml")

training_data, training_labels = get_training_sets()
fishface = create_fishface()
train_classifier(fishface, training_data, training_labels)
save_classifier()
test_classifier(fishface)
