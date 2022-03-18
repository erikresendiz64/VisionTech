from os import mkdir
import cv2
import time
import mediapipe as mp
import FDmodule

name = input('Enter Name: ')
FD = FDmodule.FaceDetector(0.75)
cam = cv2.VideoCapture(0)
pTime = 0

print("\n[INFO] Stand in the camera's view")
numImgs = 0
while True:
    ret, frame = cam.read() #read each frame, return true if a frame exists
    frame, bounds = FD.findFaces(frame)
    numImgs += 1
    cv2.imshow("Running", frame)
    if numImgs % 50 == 0:
        cv2.imwrite("Data/" + str(name) + str(int(numImgs/50)) + ".jpg", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        break
    elif numImgs >= 100: # Take 30 face sample and stop video
        break