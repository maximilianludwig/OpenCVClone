import os
import ast
import json
import Constants
import numpy as np
from PIL import Image

def pixels_to_image(pixel_str, separator=" "):
    return Image.fromarray(np.array(pixel_str.split(separator), dtype=np.uint8).reshape(Constants.IMG_SIZE,Constants.IMG_SIZE))

def image_to_pixel_array(file_path):
    im = Image.open(file_path, 'r')
    return np.array(im.getdata(), dtype=np.float32).reshape(Constants.IMG_SIZE, Constants.IMG_SIZE, 1)

def save_all_images():
    if os.path.isdir(Constants.PICTURE_FOLDER):
        print('ERROR: Directory',Constants.PICTURE_FOLDER, 'already exist. Delete manually for saving the emotion images.')
        return
    else:
        os.mkdir(Constants.PICTURE_FOLDER)
        for emotion in Constants.EMOTIONS:
            os.mkdir(os.path.join(Constants.PICTURE_FOLDER, emotion))

    for index, row in Constants.RAW_DATA.iterrows():
        img = pixels_to_image(row[Constants.COLUMN_PIXELS])
        img.save(os.path.join(Constants.PICTURE_FOLDER,str(Constants.NUMBER_TO_EMOTION[int(row[Constants.COLUMN_EMOTION])]), str(index+1) + ".png"))     

def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Wrong input, please answer!")

def list_to_dict(list):
    return {i:list[i] for i in range(len(list))}

def calculate_accuracy(n, trues):
    return trues/n

def print_vars():
    print("BATCH_SIZE:", Constants.BATCH_SIZE, ", EPOCHS:", Constants.EPOCHS)

def load_prediction_dict(model_dir):
    file_path = os.path.join(Constants.MODEL_FOLDER, model_dir, Constants.MODEL_METADATA_FILE)
    if not os.path.isfile(file_path):
        print("Prediction dictionary could not be loaded because", file_path, "File does not exist.")
        return None
    file = open(file_path, "r")
    prediction_dict = ast.literal_eval(file.read())
    file.close()
    return prediction_dict

def save_prediction_dict(model_dir, prediction_dict):
    file_path = os.path.join(Constants.MODEL_FOLDER, model_dir, Constants.MODEL_METADATA_FILE)
    if not os.path.isfile(file_path):
        file = open(file_path, "w")
        file.write(str(prediction_dict))
        file.close()

def load_history(model_dir):
    file_path = os.path.join(Constants.MODEL_FOLDER, model_dir, "history.json")
    if os.path.isfile(file_path):
        file = open(file_path, "r")
        history = json.load(file)
        file.close()
        return history

def save_history(model_dir, history):
    file_path = os.path.join(Constants.MODEL_FOLDER, model_dir, "history.json")
    file = open(file_path, "w")
    file.write(str(history).replace("'", "\""))
    file.close()

def count_files(folder):
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])