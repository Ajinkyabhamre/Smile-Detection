import numpy as np
import cv2


# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_Full_Body = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#choose an image to detect faces
#img = cv2.imread('RDJ.jpg')

#To capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:


 successful_frame_read, frame = webcam.read()

 # must convert to grayscale
 grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 #Detect faces
 face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
 body_coordinates = trained_Full_Body.detectMultiScale(grayscaled_img, scaleFactor=1.2, minNeighbors=3 )
 #Draw rectangle around the face
 for (x, y, w, h) in face_coordinates:
   cv2.rectangle(frame, (x,y),(x+w, y+h), (0, 255, 0), 2)

 for (x, y, w, h) in body_coordinates:
   cv2.rectangle(frame, (x,y),(x+w, y+h), (50, 50, 200), 4)  

 cv2.imshow('Face Detector', frame)

 key = cv2.waitKey(1) 
 if key==81 or key==113:
   break 

webcam.release()

 
"""
#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#print(face_coordinates)

#Draw rectangle around the face
for (x, y, w, h) in face_coordinates:
#[x, y, w, h] = face_coordinates[0]
  cv2.rectangle(img, (x,y),(x+w, y+h), (0, 255, 0), 2)


cv2.imshow('Face Detector', img)
cv2.waitKey() 
"""
print('Hello World')