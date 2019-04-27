"""
define emotions on the image
"""

import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

from keras.models import model_from_json
from keras.metrics import categorical_accuracy

from utils.extractor import Extractor


MODEL_FILE = '../models/model_deep.json'

WEIGHTS_FILE = '../models/weights/complex_20_cc_2.h5'

IMAGE_FILE = '../data/original/test2.jpg'

TARGET_DIR = '../images/'

MAPPING = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


def convert_to_rgb(image):
    """
    converts image to it's natural colormap
    :param image:
    :return:
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def read_model_from_disk(model_file: str, weights_file: str):
    """
    read model from json file and weights from .h5 file
    :param model_file:
    :param weights_file:
    :return:
    """
    json_file = open(model_file, 'r')

    loaded_model = json_file.read()

    model = model_from_json(loaded_model)

    model.load_weights(weights_file)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])

    return model


MODEL = read_model_from_disk(model_file=MODEL_FILE,
                             weights_file=WEIGHTS_FILE)


def process_image(image: str):
    """
    generalized function to process and label image
    :param image: path to image
    :return:
    """
    coords_path = Extractor.extract_faces(image, 1, )

    df = pd.read_csv(coords_path)

    images_locations = df.image_location.tolist()

    df['emotion'] = None

    for index, location in enumerate(images_locations):

        img = cv.imread(location,  cv.IMREAD_GRAYSCALE)

        reshaped = img.reshape((1, 48, 48, 1))

        emotion = MAPPING[MODEL.predict_classes([reshaped])[0]]

        df.at[index, "emotion"] = emotion

    original = cv.imread(image)

    for i in range(len(df)):

        cv.rectangle(original,
                     (df.loc[i, 'x_lo'], df.loc[i, 'y_lo']),
                     (df.loc[i, 'x_hi'], df.loc[i, 'y_hi']),
                     (0, 255, 0),
                     original.shape[0] // 1000)

        cv.putText(original,
                   df.loc[i, 'emotion'],
                   (df.loc[i, 'x_lo'] + 5, df.loc[i, 'y_hi'] - 5),
                   cv.FONT_HERSHEY_PLAIN,
                   original.shape[0] / 1000,
                   (0, 255, 0),
                   original.shape[0] // 400)

    plt.imsave(TARGET_DIR + image.split("/")[-1].split('.')[0] + '_labeled.jpg', convert_to_rgb(original))


process_image(IMAGE_FILE)
