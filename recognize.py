import cv2
import numpy as np
import os
import FDmodule as FD

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ["Erik", "Georgi", "Anas", "Ricardo"]

cam = cv2.VideoCapture(0)
FD = FD.FaceDetector(0.75)
print("\n [INFO] Stand in the camera's view")

while True:
    ret, frame = cam.read() #read each frame, return true if a frame exists
    frame, bounds = FD.findFaces(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if(len(bounds) != 0):
        x1,y1 = bounds[0][0], bounds[0][1]
        x2, y2 = x1 + bounds[0][2], y1 + bounds[0][3]
        id, confidence = recognizer.predict(gray[y1:y1+y2,x1:x1+x2])
        id = names[id]
        cv2.putText(
                frame, 
                str(id), 
                (x1+5,y1-5), 
                font, 
                1, 
                (255,255,255), 
                2
        )
    else:
        pass
    cv2.imshow("Running", frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        break


print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
