import cv2
import os
import numpy as np
from PIL import Image
import re
import time

def FindNumInString(str):
    findNum = re.findall('[0-9]+', str)
    num = int(findNum[0])

    return num

path = 'Dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()


def getImagesLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        paths = imagePath.split("/")
        info = paths[1]
        if not info.startswith('.'):
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            cv2.imshow("img", PIL_img)
            cv2.waitKey(0)
            img_numpy = np.array(PIL_img,'uint8')
            face = FindNumInString(info)
            faceSamples.append(img_numpy) #CHECK THIS
            ids.append(face)
        else:
            pass
    
    return faceSamples,ids
        
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
start = time.time()
faces, ids = getImagesLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml') 
end = time.time()
total = end - start
print(f"Done in {total} seconds")
print("\n [INFO] {0} face(s) trained. Exiting Program".format(len(np.unique(ids))))