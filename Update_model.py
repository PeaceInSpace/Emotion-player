import cv2
import glob
import random
import numpy as np


fishface = cv2.face.FisherFaceRecognizer_create()

data = {}


def make_sets(emotions):
    training_data = []
    training_labels = []

    for emotion in emotions:
        training = training = glob.glob("dataset\\%s\\*" %emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

    return training_data, training_labels


def run_recognizer(emotions):
    training_data, training_labels = make_sets(emotions)
   #print("training data"+str(training_data))

    print("training fisher face classifier")
    print("size of training set is: " + str(len(training_labels)) + " images")
    fishface.train(training_data, np.array(training_labels))
    print("trained")


def update(emotions):
    run_recognizer(emotions)
    print("saving model...")
    fishface.save("trained_emotion_classifier.xml")
    print("model updated and saved.")

