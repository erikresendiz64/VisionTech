import cv2 as cv
from cv2 import WINDOW_AUTOSIZE
import tensorflow
import matplotlib
import face_recognition

cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('face_detector.xml')

cv.namedWindow("Python Cam Screenshot")

name = 'Erik'
img_counter = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    
    for (x, y, w, h) in faces: 
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv.imshow("Running" , frame)

# for i in range(img_counter):
#     image = face_recognition.load_image_file("dataset/"+ name +"/image_{}.jpg".format(i))
#     face_locations = face_recognition.face_locations(image)
#     print(face_locations)

cam.release()
cam.destroyAllWindows()