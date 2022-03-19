import os
import glob
import re
from os import mkdir
import cv2
import time
import mediapipe as mp
import FDmodule
import pickle


with open('DS.pickle', 'rb') as f:
    try:
        facesList = pickle.load(f)
        print(facesList[len(facesList) - 1])
        lastFace = re.findall('[0-9]+', facesList[len(facesList) - 1])
        last = int(lastFace[0])
        faceNum = last + 1
        print(faceNum)
    except EOFError:
        facesList = []
        faceNum = 0
        print("Error")

print(f"len of faces: {len(facesList)}")

FD = FDmodule.FaceDetector(0.75)
cam = cv2.VideoCapture(0)
pTime = 0
if not os.path.exists(f'./Data/face{faceNum}'): #if face not yet in Data folder, 
    mkdir(f"./Data/face{faceNum}")
    numImgs = 0
else:
    dir = os.listdir(f'./Data/face{faceNum}') #else it does exist, list directory
    if len(dir) == 0: #if there are no images
        numImgs = 0 #start at 0 images
    else:
        list_of_files = glob.glob(f'./Data/face{faceNum}/*jpg') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        parsed = latest_file.split('/')
        numLoc = parsed[3]
        imgNum = re.findall('[0-9]+', numLoc)
        imgs = int(imgNum[0])

        numImgs = int(imgNum[0]) * 50 #to avoid overwriting previous images ^^^


print("\n[INFO] Stand in the camera's view")
while True:
    ret, frame = cam.read() #read each frame, return true if a frame exists
    frame, bounds = FD.findFaces(frame)
    numImgs += 1
    if numImgs % 50 == 0:
        cv2.imwrite(f"./Data/face{faceNum}/face{faceNum}.{int(numImgs/50)}.jpg", frame)
    cv2.imshow("Running", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        break
    elif numImgs >= 100: 
        break

faceToAdd = f'face{faceNum}'
facesList.append(faceToAdd)
with open('DS.pickle', 'wb') as f:
    pickle.dump(facesList, f)