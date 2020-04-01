from __future__ import print_function
import cv2
import os
import time
import requests
from requests.exceptions import HTTPError
from pathlib import Path
import numpy as np
import json
from IPython.display import clear_output
from collections import deque
from io import BytesIO
import threading

def is_pokerface(data, tolerance=0.1):
    avgs = np.mean(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    for i in range(len(data[0])):
        if avgs[i] - mins[i] > tolerance or maxs[i] - avgs[i] > tolerance:
            return False
    return True

def json_to_list(json):
    return [ float(json[str(x)]) for x in range(len(json)) ]

def post_picture(img):
    endpoint = "http://193.196.53.156/continuous-check"
    try:
        ret, buf = cv2.imencode('.png', img)
        post_format = BytesIO(buf)
        file = { 'image': post_format }
        response = requests.post(endpoint, files = file)
        return response 
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return None

def show_video_stream(cam):
    cv2.namedWindow("Pokerface")
    while True: 
        ret, frame = cam.read()
        if not ret:
            print("Returned show_video_stream")
            break 
        k = cv2.waitKey(1)
        if k%256 == 27:
            break
        cv2.putText(frame,str(pokerface),(frame.shape[1] - 100, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.imshow("Pokerface", frame)
    global kill_process
    kill_process = True
    cv2.destroyAllWindows()
    cam.release()

def pokerface_handler(cam, interval = 2, buffer = 3):
    emotion_array = deque( [ [0, 0, 0, 0, 0, 0] for i in range(buffer) ] )
    global kill_process
    while kill_process==False:
        ret, frame = cam.read()
        if not ret:
            print("Returned pokerface_handler")
            break 
        response = post_picture(frame)
        if response.status_code != 200:
            continue
            
        emotion_values = json_to_list(json.loads(response.content))
        emotion_array.appendleft(emotion_values)
        emotion_array.pop()

        global pokerface
        if is_pokerface(emotion_array, 0.1):
            #clear_output(wait=True)
            print("Pokerface = True")
            pokerface = True
        else:
            #clear_output(wait=True)
            print("Pokerface = False")
            pokerface = False
        time.sleep(interval)
    cam.release()
    print("pokerface_handler finished")

kill_process = False
pokerface = False
def start_client(interval, buffer):
    cam = cv2.VideoCapture(0)
    pokerface_thread = threading.Thread(target=pokerface_handler, args=(cam, interval, buffer,), daemon=False)
    pokerface_thread.start()
    show_video_stream(cam)
    pokerface_thread.join()