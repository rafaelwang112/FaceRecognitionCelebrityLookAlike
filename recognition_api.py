import cv2
import numpy as np
import face_recognition
import os
import json

path = 'data'
encodings_path = os.path.join(path, "encodings.npy")
names_path = os.path.join(path, "names.json")

if not os.path.exists(encodings_path) or not os.path.exists(names_path):
    print ("Run encode_faces.py first.")
    exit()

my_encode_list = np.load(encodings_path)
with open (names_path, "r") as file:
    names = json.load(file)

capture = cv2.VideoCapture(1)
if not capture.isOpened():
    print ("Could not open camera.")
    exit()
while True:
    suc, img = capture.read()
    img_shrink = cv2.resize(img,(0,0),None,0.20,0.20)
    img_shrink = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2RGB)

    #for multiple faces in frame
    faces_loc = face_recognition.face_locations(img_shrink)
    encode_imgs = face_recognition.face_encodings(img_shrink, faces_loc)

    #find matches
    for encoded_face, face_loc in zip(encode_imgs, faces_loc):
        match = face_recognition.compare_faces(my_encode_list, encoded_face)
        dist = face_recognition.face_distance(my_encode_list, encoded_face)
        best_index = np.argmin(dist)

        if match[best_index]:
            name = names[best_index]
            y1,x2,y2,x1 = face_loc
            y1,x2,y2,x1 = y1*5, x2*5, y2*5, x1*5
            cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2),(255,0,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),3)
            
    cv2.imshow('Webcam', img)
    if (cv2.waitKey(1) & 0xFF == 13):
        break

capture.release()
