from emotion_network import EmotionNeuronet
import sys

train_classifier = 'cascade'
continue_train = False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for flag in sys.argv:
            if flag == '-yolo':
                train_classifier = 'yolo'
            elif flag == 'cascade':
                train_classifier = 'cascade'

            if flag == '-continue':
                continue_train = True

network = EmotionNeuronet(fromload=continue_train, load_dataset=True, train_classifier=train_classifier)
network.train()

