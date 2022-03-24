import os
import glob
import re
from os import mkdir
import cv2
import time
import mediapipe as mp
import FDmodule
import pickle
import face_recognition

def FindNumInString(str):
    findNum = re.findall('[0-9]+', str)
    num = int(findNum[0])

    return num

def Face(file):
    with open('faces.pickle', 'rb') as f:
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
                cv2.imwrite(f"./Dataset/face{faceNum}.{int(imgsInDir/5)}.jpg", frame)
            pass
        cv2.imshow("Running", frame)
        

        k = cv2.waitKey(1)

        if k % 256 == 27:
            break
        elif imgsInDir >= 20: 
            break

def StoreEncodings(images_path, list_of_names):
    encodings = []
    names = []
    images_path = glob.glob(os.path.join(images_path, "*.*"))
    print("{} encoding images found.".format(len(images_path)))
    img_encoded = 0

    # Store image encoding and names
    for img_path in images_path:
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        paths = img_path.split("/")
        info = paths[1]
        if not info.startswith('.'):
            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            idx = FindNumInString(filename)
            name = list_of_names[idx]
            # Get encoding
            if face_recognition.face_encodings(rgb_img):
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                img_encoded += 1
                encodings.append(img_encoding)
                names.append(name)
                print("image succesfully encoded")
            else:
                print("encoding unsuccesful")
        else:
            pass
        
    print(f'{img_encoded} images encoded')
    print("\nEncoding images loaded")

    return encodings, names

facesList, faceNum = Face('faces.pickle')
FD = FDmodule.FaceDetector()
cam = cv2.VideoCapture(0)
imgsInDir = Directory()

print("\n[INFO] Stand in the camera's view")
StoreData(cam, faceNum, imgsInDir)

#keep track of face
faceToAdd = f'face{faceNum}' #update Pickle File
facesList.append(faceToAdd)
with open('faces.pickle', 'wb') as f:
    pickle.dump(facesList, f)

#load encodings
Names = ['Erik', 'Iron Man', 'Ryan Reynolds']
print("[INFO] serializing encodings...")
knownEncodings, knownNames = StoreEncodings(f'Data/face{faceNum}/', Names)
with open('encodings.pickle', 'rb') as f:
    try:
        data = pickle.load(f)
        _enc = data["encodings"]
        _names = data["names"]
        for i in range(len(knownEncodings)):
            _enc.append(knownEncodings[i])
            _names.append(knownNames[i])
        data = {"encodings": _enc, "names": _names}
        with open('encodings.pickle', 'wb') as f:
            pickle.dump(data, f)
    except EOFError:
        data = {"encodings": knownEncodings, "names": knownNames}
        with open('encodings.pickle', 'wb') as f:
            pickle.dump(data, f)
