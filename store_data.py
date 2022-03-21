import os
import glob
import re
from os import mkdir
import cv2
import time
import mediapipe as mp
import FDmodule
import pickle

def FindNumInString(str):
    findNum = re.findall('[0-9]+', str)
    num = int(findNum[0])

    return num

def Face(file):
    with open('Pickle.pickle', 'rb') as f:
        try:
            facesList = pickle.load(f)
            lastFace = facesList[len(facesList) - 1]
            numInStr = FindNumInString(lastFace)
            faceNum = numInStr + 1
        except EOFError:
            facesList = []
            faceNum = 0
        
        return facesList, faceNum

def Directory():
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
            numInStr = FindNumInString(numLoc)
            numImgs = numInStr #to avoid overwriting previous images ^^^
    return numImgs

def StoreData(cam, faceNum, imgsInDir):
    while True:
        ret, frame = cam.read() #read each frame, return true if a frame exists
        frame, bounds = FD.findFaces(frame)

        imgsInDir += 1
        if(len(bounds) != 0):
            x1,y1 = bounds[0][0], bounds[0][1]
            x2, y2 = x1 + bounds[0][2], y1 + bounds[0][3]
            if imgsInDir % 5 == 0:
                cv2.imwrite(f"./Data/face{faceNum}/face{faceNum}.{int(imgsInDir/5)}.jpg", frame)
                cv2.imwrite(f"./Dataset/face{faceNum}.{int(imgsInDir/5)}.jpg", frame[y1:y2, x1:x2])
        else:
            pass
        cv2.imshow("Running", frame)
        

        k = cv2.waitKey(1)

        if k % 256 == 27:
            break
        elif imgsInDir >= 25: 
            break

facesList, faceNum = Face('Pickle.pickle')
FD = FDmodule.FaceDetector()
cam = cv2.VideoCapture(0)
imgsInDir = Directory()

print("\n[INFO] Stand in the camera's view")
StoreData(cam, faceNum, imgsInDir)

faceToAdd = f'face{faceNum}' #update Pickle File
facesList.append(faceToAdd)
with open('Pickle.pickle', 'wb') as f:
    pickle.dump(facesList, f)