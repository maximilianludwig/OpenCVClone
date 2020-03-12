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

# This method can be used for creating the training, validation and testing sets.<br>
# ***@Param usage*** should be **"Training"**, **"PublicTest"** or **"PrivateTest"**<br>
# ***@Param emotions*** should be a set of emotions which you want to filter
# 
# *num_list* is a list of the numbers which are used in the database to represent the emotions<br>
# *data_set* is the set of all filtered data rows of the used dataset
# *x_set*    is the array of all filtered pixel representations
# *y_set*    is the array of all expected emotions as numbers which can differ from number representation of the used database
def create_set(usage, emotions=Constants.EMOTIONS):
    num_list = list(map(lambda emotion: Constants.EMOTION_TO_NUMBER[emotion], emotions))
    data_set = Constants.RAW_DATA[Constants.RAW_DATA[Constants.COLUMN_USAGE].isin(usage) & Constants.RAW_DATA[Constants.COLUMN_EMOTION].isin(num_list)]
    x_set = np.array(list(map(lambda x: np.array(x.split(" "), dtype=np.float32).reshape(48,48,1), data_set[Constants.COLUMN_PIXELS])))
    y_set = np.array(list(map(lambda x: emotions.index(Constants.NUMBER_TO_EMOTION[x]), data_set[Constants.COLUMN_EMOTION].values)))
    return x_set, y_set

# This method can be ussed for creating a model with the given layers.<br>
# ***@Param name*** is the name of the model which you need for saving it later<br>
# ***@Param emotions*** is a set which contains all possible output emotions<br>
# 
# *model* is the global variable which is used for training, saving and testing
# *prediction_dict* is the dictionary which translate the used emotion numbers to the emotion, in case you don't use all emotions of database you can't use the *Constants.EMOTION_TO_NUMBER* so the translation is stored individually<br>
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

# This method is used for training the model with given data.<br>
# ***@Param x_train*** is a set of pixels from which the model should learn<br>
# ***@Param y_train*** is a set of expected outputs to the corresponding pixel input<br>
def train(x_train, y_train):
    global history
    history = model.fit(x_train, y_train, epochs=Constants.EPOCHS, batch_size=Constants.BATCH_SIZE).history
    print(model.summary())

# This method is used for training the model with given data and validate on each epoch.<br>
# ***@Param x_train*** is a set of pixels from which the model should learn<br>
# ***@Param y_train*** is a set of expected outputs to the corresponding pixel input<br>
# ***@Param x_validate*** is a set of pixels from which the accuracy can be validated on each epoch<br>
# ***@Param y_validate*** is a set of expected outputs to the corresponding pixel input in *x_validate*<br>
def train_and_validate(x_train, y_train, x_validate, y_validate):
    global history
    history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), shuffle=True, epochs=Constants.EPOCHS, batch_size=Constants.BATCH_SIZE).history
    print(model.summary())

# This method is used for graphical visualization of accuracy statistics from the training process
def visualize_accuracy_history():
    plt.plot(history['accuracy'], "bo")
    plt.plot(history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# This method is used for graphical visualization of loss statistics from the training process
def visualize_loss_history():
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# This method is used for testing the previously trained model.<br>
# ***@Param x_test*** is a set of pixels for which the model should predict an emotion as number<br>
# ***@Param y_test*** is a set of expected outputs to the corresponding pixel input<br>
# 
# *arr* is a two dimensional array which stores the counts of each emotion and the counts of right predictions of each emotion which you will need to calculate accuracy and some other stats<br>
# 
# You can translate the predicted number to an emotion with the in **create_model** defined *prediction_dict*
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

# This method is used for saving a model as an directory which contains all information.
# All models are saved in a model folder which will be created if not exist.<br>
# In this model folder the model will be saved with the ending **.model**<br>
# If a model with same name already exist you will be ask for confirming.
# At last the individually generated *prediction_dict* will be saved into the saved model directory.
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

# This model is used for loading an previously saved model.
# The *model* and *prediction_dict* will be extracted and stored into the global variables.
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

# This method is be used for creating a full lifecycle of a model from creating over training (and validating) to testing and saving including all required sets.<br>
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





