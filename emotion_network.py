from dataset_class import Dataset

import numpy as np
from os.path import join, isfile
from config import *
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import glob


class EmotionNeuronet:

    def __init__(self, fromload=True, load_dataset=True, train_classifier='cascade'):
        if load_dataset:
            self.dataset = Dataset(train_classifier=train_classifier)
        self.network = input_data(shape=[None, FACE_SIZE, FACE_SIZE, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = local_response_normalization(self.network)
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')
        self.network = regression(self.network, optimizer='momentum', loss='categorical_crossentropy')
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path=CHECKPOINT_PATH,
            best_checkpoint_path=CHECKPOINT_PATH,
            tensorboard_verbose=2
        )

        if fromload:
            self.load()

    def train(self):

        print('Start training...')
        self.model.fit(
            self.dataset.images, self.dataset.labels,
            validation_set=(self.dataset.images_test,
                            self.dataset.labels_test),
            n_epoch=100,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=200,
            snapshot_epoch=True,
            run_id='emotion_recognition'
        )
    def load(self):
        if len(glob.glob(join(CHECKPOINT_PATH, MODEL_DEFAULT) + '.*'))!=0:
            self.model.load(join(CHECKPOINT_PATH, MODEL_DEFAULT))
            print('Successfully loaded model from ' + MODEL_DEFAULT)
        else:
            print("No model with name {} found in {}!".format(MODEL_DEFAULT, CHECKPOINT_PATH))

    def predict(self, image):

        if image is None:
            return None
        print("Network gets {}".format(image.shape))
        image = image.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        print("Network gets {}".format(image.shape))
        return self.model.predict(image)


#network = EmotionNeuronet()
#network.train()