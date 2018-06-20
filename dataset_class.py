import os
import numpy as np
import pandas as pd
from os.path import join, isfile, isdir
from config import *
from fer2013_converter import Converter
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from PIL import Image


class Dataset:

    def __init__(self, train_classifier='cascade'):
        if numpy_dataset_exists():
            self.load()
        elif fer_dataset_exists():
            while True:
                answer = input("Could not find numpy dataset in {}\nDo you "
                      "want to build a new dataset from FER2013 database?\n"
                      "it may last up to 3 hours \n"
                      "[Y/n]:".format(join(os.getcwd(), DATASET_NP_PATH)))
                if answer in ['y', 'Y', 'n', 'N', 'q', 'Q']:
                    break
            if answer in ['y', 'Y']:
                self.build(train_classifier=train_classifier)
            else:
                exit()
        else:
            print("Could not find any dataset!")
            exit()

    # Set include_unknown=True to include "unknown" emotion in dataset
    # Set only_one_activation=True to eclude labels with float activations
    #                                                    (for example 0.5)

    def build(self, include_unknown=False, only_one_activation=True, train_classifier='cascade'):
        print("Building dataset from fer2013.csv and fer2013plus.csv")
        print("wait...")
        label_data = pd.read_csv(join(DATASET_CSV_PATH, FER_LABEL))
        image_data = pd.read_csv(join(DATASET_CSV_PATH, FER_IMAGE))

        # Prepare label data
        label_data = label_data.values[:, 2:]
        label_data -= np.amax(label_data, axis=1, keepdims=True) - 1
        label_data = label_data.clip(min=0)
        label_data = normalize(label_data, norm='l1')

        # Lists for processed data
        labels = []
        images = []
        convert = Converter(train_classifier)

        for index, row in image_data.iterrows():

            append = True
            emotion = label_data[index]
            image = convert.string_to_image(row['pixels'])

            if emotion[-1] == 1:
                append = False
            if image is None:
                append = False
            if append and (not include_unknown) and emotion[-2] == 1:
                append = False
            if append and only_one_activation and np.count_nonzero(emotion) != 1:
                append = False

            if append:
                labels.append(emotion[:len(emotion) - 1])
                images.append(image)

            index += 1
            print("Images processed: {}/{}".format(index, label_data.shape[0]))

        print("\n\n\nSuccess!\n\nSaving dataset...")

        labels = np.array(labels)
        images = np.array(images)
        images = images.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        labels = labels.reshape([-1, len(EMOTIONS)])

        self.images, self.images_test, self.labels, self.labels_test \
            = train_test_split(images, labels, test_size=0.20, random_state=42)

        if not isdir(DATASET_NP_PATH):
            os.makedirs(DATASET_NP_PATH)
        np.save(join(DATASET_NP_PATH, SHUFFLE_IMAGE), self.images)
        np.save(join(DATASET_NP_PATH, SHUFFLE_IMAGE_TEST), self.images_test)
        np.save(join(DATASET_NP_PATH, SHUFFLE_LABEL), self.labels)
        np.save(join(DATASET_NP_PATH, SHUFFLE_LABEL_TEST), self.labels_test)
        print("shuffled labels and images have been saved in {}!".format(DATASET_NP_PATH))

    def load(self):
        self.images =      np.load(join(DATASET_NP_PATH, SHUFFLE_IMAGE))
        self.images_test = np.load(join(DATASET_NP_PATH, SHUFFLE_IMAGE_TEST))
        self.labels =      np.load(join(DATASET_NP_PATH, SHUFFLE_LABEL))
        self.labels_test = np.load(join(DATASET_NP_PATH, SHUFFLE_LABEL_TEST))
        print("Dataset has been successfully loaded!")


def numpy_dataset_exists():
    if (
            isfile(join(DATASET_NP_PATH, SHUFFLE_IMAGE)) and
            isfile(join(DATASET_NP_PATH, SHUFFLE_IMAGE_TEST)) and
            isfile(join(DATASET_NP_PATH, SHUFFLE_LABEL)) and
            isfile(join(DATASET_NP_PATH, SHUFFLE_LABEL_TEST))
    ):
        return True
    else:
        return False


def fer_dataset_exists():
    if (
            isfile(join(DATASET_CSV_PATH, FER_LABEL)) and
            isfile(join(DATASET_CSV_PATH, FER_IMAGE))
    ):
        return True
    else:
        return False
