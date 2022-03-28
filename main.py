from store_data import Store
from face_rec import SimpleFacerec
import FDmodule
import pickle
import cv2
import os
import glob

cmd = input('Enter Command: ')
while cmd != 'Quit':
    if cmd == 'Store':
        FD = FDmodule.FaceDetector(0.75)
        SD = Store(FD)
        facesList, faceNum = SD.Face('faces.pickle')
        cam = cv2.VideoCapture(0)
        imgsInDir = SD.Directory()

        print("\n[INFO] Stand in the camera's view")
        SD.StoreData(cam, faceNum, imgsInDir)

        #keep track of face
        faceToAdd = f'face{faceNum}' #update Pickle File
        facesList.append(faceToAdd)

        #load encodings
        print("[INFO] serializing encodings...")
        faceAdded, hasEncodings = SD.StoreEncodings(f'Data/face{faceNum}/')

        if (hasEncodings):
            with open('faces.pickle', 'wb') as f:
                pickle.dump(facesList, f)

            with open('encodings.pickle', 'rb') as f:
                try:
                    dictFaces = pickle.load(f)
                    dictFaces[faceToAdd] = faceAdded
                    with open('encodings.pickle', 'wb') as f:
                        pickle.dump(dictFaces, f)
                except EOFError:
                    dictFaces = {}
                    dictFaces[faceToAdd] = faceAdded
                    with open('encodings.pickle', 'wb') as f:
                        pickle.dump(dictFaces, f)
        else:
            print("Sorry, Could Not Encode Face. Please Try Again")
            files = glob.glob(f'./Data/face{faceNum}/*.jpg', recursive=True)
            try:
                for f in files:
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    elif (cmd == 'Recognize'):
        with open('encodings.pickle', 'rb') as f:
            dictFaces = pickle.load(f)
        sfr = SimpleFacerec(dictFaces)

        while True:
            ret, frame = cam.read()
        # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame, f'User: {name}' ,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        
    cmd = input('Enter Command: ')