import os
import ast
import json
import pandas
import Constants
import numpy as np
import matplotlib.pyplot as plt

from lib import list_to_dict, save_prediction_dict, load_prediction_dict, save_history, load_history, calculate_accuracy, yes_or_no
from PIL import Image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

model = None
history = None
prediction_dict = None

def create_set(usage, emotions=Constants.EMOTIONS):
    num_list = list(map(lambda emotion: Constants.EMOTION_TO_NUMBER[emotion], emotions))
    data_set = Constants.RAW_DATA[Constants.RAW_DATA[Constants.COLUMN_USAGE].isin(usage) & Constants.RAW_DATA[Constants.COLUMN_EMOTION].isin(num_list)]
    x_set = np.array(list(map(lambda x: np.array(x.split(" "), dtype=np.float32).reshape(48,48,1), data_set[Constants.COLUMN_PIXELS])))
    y_set = np.array(list(map(lambda x: emotions.index(Constants.NUMBER_TO_EMOTION[x]), data_set[Constants.COLUMN_EMOTION].values)))
    return x_set, y_set

def create_model(name, emotions=Constants.EMOTIONS):
    global model
    global prediction_dict
    prediction_dict = list_to_dict(emotions)
    model = Sequential(name=name)
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(emotions), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def train(x_train, y_train):
    global history
    history = model.fit(x_train, y_train, epochs=Constants.EPOCHS, batch_size=Constants.BATCH_SIZE).history
    print(model.summary())

def train_and_validate(x_train, y_train, x_validate, y_validate):
    global history
    history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), shuffle=True, epochs=Constants.EPOCHS, batch_size=Constants.BATCH_SIZE).history
    print(model.summary())

def visualize_accuracy_history():
    plt.plot(history['accuracy'], "bo")
    plt.plot(history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def visualize_loss_history():
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def test(x_test, y_test):
    n = len(y_test)
    arr = [ [0,0] for i in range(len(prediction_dict.values())) ]
    predictions = model.predict(x_test)
    for i in range(len(y_test)):
        arr[y_test[i]][0] += 1
        if y_test[i] == np.argmax(predictions[i]):
            arr[y_test[i]][1] += 1
    print("N:", n, "True:", sum([x[1] for x in arr]), "False:", sum([x[0] - x[1] for x in arr]), "Average:", calculate_accuracy(n, sum([x[1] for x in arr])))
    print("Accuracies for given emotions:")
    print(list(map(lambda x: calculate_accuracy(x[0], x[1]), arr)))

def save():
    if not os.path.isdir(Constants.MODEL_FOLDER):
        os.mkdir(Constants.MODEL_FOLDER)
    model_dir = model.name + ".model"
    if os.path.isdir(os.path.join(Constants.MODEL_FOLDER, model_dir)):
        print("WARNING: " + model_dir + " already exists")
        if not yes_or_no("Do you really want to override existing model?"):
            print("Model was NOT saved!")
            return
    model.save(os.path.join(Constants.MODEL_FOLDER, model_dir))
    save_prediction_dict(model_dir, prediction_dict)
    if history is not None:
        save_history(model_dir, history)
    print("Model saved to " + os.path.join(Constants.MODEL_FOLDER, model_dir))

def load(model_dir):
    global model
    global prediction_dict
    global history
    if not os.path.isdir(Constants.MODEL_FOLDER):
        print("Model folder", Constants.MODEL_FOLDER, "does not exist! Model can not find model.")
        return
    model = load_model(os.path.join(Constants.MODEL_FOLDER, model_dir))
    prediction_dict = load_prediction_dict(model_dir)
    history = load_history(model_dir)
    print("Model",model_dir,"was loaded")

def full_run(file_name):
    emotions = Constants.EMOTIONS.copy()
    emotions.remove("Disgust")
    x_train, y_train = create_set(["Training"], emotions)
    x_validate, y_validate = create_set(["PublicTest"], emotions)
    x_test, y_test = create_set(["PrivateTest"], emotions)
    create_model(file_name, emotions)
    train_and_validate(x_train, y_train, x_validate, y_validate)
    visualize_accuracy_history()
    visualize_loss_history()
    test(x_test, y_test)
    save()

#full_run("Best")
load("Best.model")
visualize_accuracy_history()
visualize_loss_history()
emotions = Constants.EMOTIONS.copy()
emotions.remove("Disgust")
x_test, y_test = create_set(["PrivateTest"], emotions)
test(x_test, y_test)
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, dpi=96)





