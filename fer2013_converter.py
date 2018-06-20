import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from config import *
from os.path import join, isfile
from PIL import Image
import cv2
from darkflow.net.build import TFNet
import time


np.set_printoptions(threshold=np.nan)

options = {
            'model': join(YOLO_PATH, YOLO_CFG),
            'load': join(YOLO_PATH, YOLO_WEIGHTS),
            'threshold': 0.4,
            'gpu': 0.0
        }


class Converter:

    def __init__(self, classifier='cascade'):

        if classifier=='yolo':
            self.classifier = TFNet(options)
            self.classifier_name = 'yolo'
        else:
            self.classifier = cv2.CascadeClassifier(join(HAARCASCADE_PATH, HAARCASCADE))
            self.classifier_name = 'cascade'

    def find_faces(self, image, multiple=False):
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if len(image.shape) == 3:
            if image.shape[2] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        face_list = []
        if self.classifier_name == 'yolo':
            face_list_dictionary = self.classifier.return_predict(image)
            for face in face_list_dictionary:
                face = Face(
                    face['topleft']['x'],
                    face['bottomright']['x'],
                    face['topleft']['y'],
                    face['bottomright']['y']
                )
                face.calc_area()
                face_list.append(face)
        else:
            face_list_dictionary = self.classifier.detectMultiScale(
                image,
                scaleFactor=1.3,
                minNeighbors=5
            )
            for face in face_list_dictionary:
                face = Face(
                    face[0],
                    face[0]+face[3],
                    face[1],
                    face[1]+face[2]
                )
                face.calc_area()
                face_list.append(face)

        #  If no face has been found
        if not len(face_list) > 0:
            print("No face has been found")
            return None
        if not multiple:
            # Find biggest face on picture
            max_area_face = face_list[0]
            for face in face_list:
                if face.area > max_area_face.area:
                    max_area_face = face
            face_list = max_area_face
        return face_list

    def crop_face(self, image, face):

        if (image is None) or (face is None):
            print("none")
            return None
        else:
            if len(image.shape) > 2:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image.reshape(image.shape[0], image.shape[1])
            image = image[
                    face.y:face.ybottom,
                    face.x:face.xright
                    ]
            # Resize image to FACE_SIZE
            try:
                image = cv2.resize(image, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC) / 255.
            except Exception:
                print("Resize failed")
                return None
            image.reshape(image.shape[0], image.shape[1])
            return image

    def string_to_image(self, image):
        image = np.fromstring(
            str(image), dtype=np.uint8, sep=' ').reshape((FACE_SIZE, FACE_SIZE))
        image = add_grayborder(image)
        face = self.find_faces(image, multiple=False)

        image = self.crop_face(image, face)
        return image


def to_gray_add_graychannel(image):

    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    return image


def add_grayborder(image):
    image = to_gray_add_graychannel(image)
    bordersize = BORDER
    image = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[200] )

    return image


class Face:

    def __init__(self, x, xright, y, ybottom):
        self.x = x
        self.xright = xright
        self.y = y
        self.ybottom = ybottom
        self.area = (xright-x)*(ybottom-y)

    def calc_area(self):
        self.area = (self.xright - self.x) * (self.ybottom - self.y)

    def data(self):
        return self.x, self.xright, self.y, self.ybottom
