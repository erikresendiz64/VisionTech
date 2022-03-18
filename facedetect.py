import os
import glob
import re
from os import mkdir
import cv2
import time
import mediapipe as mp
import FDmodule

name = input('Enter Name: ')
FD = FDmodule.FaceDetector(0.75)
cam = cv2.VideoCapture(0)
pTime = 0
if not os.path.exists(f'./Data/{name}'):
    mkdir(f"./Data/{name}")
    numImgs = 0
else:
    dir = os.listdir(f'./Data/{name}')
    if len(dir) == 0:
        numImgs = 0
    else:
        list_of_files = glob.glob(f'./Data/{name}/*jpg') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        parsed = latest_file.split('/')
        numLoc = parsed[3]
        imgNum = re.findall('[0-9]+', numLoc)
        imgs = int(imgNum[0])

        numImgs = int(imgNum[0]) * 50


print("\n[INFO] Stand in the camera's view")
while True:
    ret, frame = cam.read() #read each frame, return true if a frame exists
    frame, bounds = FD.findFaces(frame)
    numImgs += 1
    if numImgs % 50 == 0:
        cv2.imwrite(f"./Data/{name}/{name}{int(numImgs/50)}.jpg", frame)
    cv2.imshow("Running", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        break
    elif numImgs >= 200: # Take 30 face sample and stop video
        break