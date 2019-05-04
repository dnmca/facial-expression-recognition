

import pandas as pd
import numpy as np

from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter

import random

mapping = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


def flip_horizontally(image):
    return np.flip(image, axis=1)


def add_noise(image):
    return image + np.random.poisson(image.astype('float64'))


def rotate_right(image, angle=-20):
    return rotate(image, angle, mode='reflect', reshape=False)


def rotate_left(image, angle=20):
    return rotate(image, angle, mode='reflect', reshape=False)


def add_blur(image):
    return gaussian_filter(image, sigma=1)


def find_subsets(df: pd.DataFrame, class_to_fraction: dict):
    """
    select random subsets for given classes with given fractions
    :param df: data
    :param class_to_fraction: dictionary {'emotion' : fraction of the data to that is going to be augmented}
    :return: indices of the images to augmented
    """
    data = df.loc[df['y'].isin(class_to_fraction.keys())]
    indices = []
    for emotion in class_to_fraction.keys():
        temp = data.loc[data['y'] == emotion]
        indices = indices + random.sample(temp.index.tolist(), int(class_to_fraction[emotion] * len(temp)))

    return indices


class Augmenter:

    @staticmethod
    def augment_images(df: pd.DataFrame, target_file: str, class_to_fraction: dict, augmentations: list) -> pd.DataFrame:

        # df = pd.read_csv(source_file)

        df['category'] = df['category'].str.strip()

        del df['Unnamed: 0']

        df['y'] = df['y'].map(mapping)

        indices = find_subsets(df, class_to_fraction)

        print(len(indices))
        temp = []

        if 'flip' in augmentations:
            for index in indices:
                temp.append([df.iloc[index, 0]] + flip_horizontally(
                    np.array(df.iloc[index, 1:2305]).reshape((48, 48))).flatten().tolist() + [df.iloc[index, 2305]])

        if 'noise' in augmentations:
            for index in indices:
                temp.append([df.iloc[index, 0]] + add_noise(
                    np.array(df.iloc[index, 1:2305]).reshape((48, 48))).flatten().tolist() + [df.iloc[index, 2305]])

        if 'rotate_right' in augmentations:
            for index in indices:
                temp.append([df.iloc[index, 0]] + rotate_right(
                    np.array(df.iloc[index, 1:2305], dtype='int').reshape((48, 48))).flatten().tolist() + [df.iloc[index, 2305]])

        if 'rotate_left' in augmentations:
            for index in indices:
                temp.append([df.iloc[index, 0]] + rotate_left(
                    np.array(df.iloc[index, 1:2305], dtype='int').reshape((48, 48))).flatten().tolist() + [df.iloc[index, 2305]])

        if 'blur' in augmentations:
            for index in indices:
                temp.append([df.iloc[index, 0]] + add_blur(
                    np.array(df.iloc[index, 1:2305], dtype='int').reshape((48, 48))).flatten().tolist() + [df.iloc[index, 2305]])

        print(len(temp))

        data = pd.DataFrame(temp, columns=df.columns)

        if target_file:
            data.to_csv(target_file)

        return data


# if __name__ == "__main__":
#
#     Augmenter.augment_images(source_file='../data/emotions/emotions.csv',
#                              target_file='../data/emotions/augmentations.csv',
#                              class_to_fraction={'Disgust': 1,
#                                                  'Sad': 0.1,
#                                                  'Fear': 0.1,
#                                                  'Neutral': 0.1,
#                                                  'Angry': 0.1
#                                                  },
    #                              augmentations=['flip', 'rotate_right', 'rotate_left', 'blur', 'noise'])
#



