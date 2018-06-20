import cv2
import sys
from config import *
from emotion_network import EmotionNeuronet
import numpy as np
from os.path import join
from fer2013_converter import *

classifier = 'cascade'
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-yolo':
            classifier = 'yolo'
        elif sys.argv[1] == '-cascade':
            classifier = 'cascade'

converter = Converter(classifier)
network = EmotionNeuronet(fromload=True, load_dataset=False)
video_capture = cv2.VideoCapture(0)

emoji = []

for index, emotion in enumerate(EMOTIONS):
    emoji.append(cv2.imread(join(PICTURES_PATH, emotion) + '.png', -1))

while True:
    _, frame = video_capture.read()
    # Predict
    face = converter.find_faces(frame)
    result = network.predict(converter.crop_face(frame, face))

    # Calculate coordinates for smiley
    if face is not None:
        x, x2, y, y2 =face.x, face.xright, face.y, face.ybottom
        xcenter = int((x2-x)/2)
        ycenter = int((y2-y)/2)
        half_size = min(xcenter, ycenter)
        xcenter = x+xcenter
        ycenter = y+ycenter


    # Write results on screen
    if (result is not None) and (face is not None):
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (x2 + 10, y + index * 15 + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 100, 100), 1)
            cv2.rectangle(frame, (x2 + 75, y + index * 15 + 15),
                          (x2 + 75 + int(result[0][index] * 70), y + (index + 1) * 15 + 9),
                          (0, 200, 200), -1)

        face_emoji = emoji[np.argmax(result[0])]
        if (face_emoji is not None) and (face_emoji.shape[0] > 1):
            face_emoji = cv2.resize(face_emoji, (half_size*2, half_size*2), interpolation=cv2.INTER_CUBIC)

            # Use alpha channel for transparency
            for c in range(0, 3):
                frame[ycenter-half_size:ycenter+half_size, xcenter-half_size:xcenter+half_size, c] = \
                    face_emoji[:, :, c] * (face_emoji[:, :, 3] / 255.0) + \
                    frame[ycenter-half_size:ycenter+half_size, xcenter-half_size:xcenter+half_size, c] * (1.0 - face_emoji[:, :, 3] / 255.0)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kill CV2 process
video_capture.release()
cv2.destroyAllWindows()

