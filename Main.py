import cv2
import numpy as np
import os
import time
import glob
import Update_model
import argparse
import random
from playsound import playsound
import winsound

from pygame import mixer

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)
model_face = cv2.face.FisherFaceRecognizer_create()

randomfile = random.choice(os.listdir("D:/Practicals/Sem VII/Minor Project/Minor project Final/Songs"))
file = 'D:/Practicals/Sem VII/Minor Project/Minor project Final/Songs'+ randomfile


def check(update1):

    try:
        model_face.read("D:/Practicals/Sem VII/Minor Project/Minor project Final/trained_emotion_classifier.xml")

    except:
        print('No xml file found, please type update to create or update the model.\n')
        update1 = input("HINT: enter update.\n")
    return update1

face_dict = {}

emotions = ['happy', 'sad', 'netural', 'angry','shock']

def crop_face(clahe_image, face):

    for x,y,w,h in face:
        faceSlice = clahe_image[y:y+h, x:x+w]
        faceSlice = cv2.resize(faceSlice,(40,40))

    face_dict ["face%s" %(len(face_dict)+1)] = faceSlice
    return faceSlice


def save_face_data(emotions):

    print("\n please show your  "+ emotions.isupper() +"face when timer goes off and keep stable expression.")

    for i in range(0,5):
        print(5-i)
        time.sleep(1)

    while len(face_dict)<20:

        count = 0
        detect_face()
        for value in face_dict.keys():
            count += 1
            cv2.imwrite("dataset/%s/%s.jpg" %(emotions, len(glob.glob("dataset/%s/*" %emotions))),face_dict[value])

    face_dict.clear()


def detect_face():

    ret, image = camera.read()
    image = cv2.flip(image,1)
    gray_photo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray_photo)

    face = face_detect.detectMultiScale(clahe_image, scaleFactor=1.1,minNeighbors=10,minSize=(10,10))

    for (x,y,w,h) in face:
         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


    if len(face)==1:
        face_slice = crop_face(clahe_image,face)
        cv2.imshow("withcropedface",face_slice)

    else:
        print('multiple face detected')


def folder(emotions):

    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')
    else:
        pass

    for i in emotions:
        if os.path.exists("D:/Practicals/Sem VII/Minor Project/Minor project Final/dataset/"+i):
            pass
        else:
            os.mkdir("D:/Practicals/Sem VII/Minor Project/Minor project Final/dataset/"+i)


def update_model():

    print("Your Model is updating.Please Wait..")

    folder(emotions)

    for j in range(0,len(emotions)):
        save_face_data(emotions[j])

    print("Looking Good")

    Update_model.update(emotions)

def recognize_emotion():

    predictions =[]
    confidence = []

    for i in face_dict.keys():
        #print(face_dict[i])

        predict, confid = model_face.predict(face_dict[i])
        cv2.imwrite("images\\%s.jpg" %i,face_dict[i])
        predictions.append(predict)
        confidence.append(confid)

    y=emotions[max(set(predictions), key=predictions.count)]

    print("I think you're %s" %y)
    #print("prediction",predictions)

    if(y=="happy"):
       playsound(file,winsound.SND_FILENAME)
    elif(y=="sad"):
       playsound(file,winsound.SND_FILENAME)
    elif(y=="angry"):
       playsound(file,winsound.SND_FILENAME)
    elif(y == "neutral"):
        playsound(file, winsound.SND_FILENAME)





while True:

    update1= "Not"
    update1 = check(update1)

    detect_face()

    if update1 == 'update':
        update_model()
        break

    elif len(face_dict)==20:
        detect_face()
        recognize_emotion()
        break
