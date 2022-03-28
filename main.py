from store_data import Store
from face_rec import SimpleFacerec
import FDmodule
import pickle
import cv2
import os
import glob
from time import *
import PySimpleGUI as sg

def StoreUser():
    # print(time())
    print("New User, please step in front of the camera.")
    sleep(5)
    # print(time())
    cam = cv2.VideoCapture(0)
    FD = FDmodule.FaceDetector(0.75)
    SD = Store(FD)
    facesList, faceNum = SD.Face('faces.pickle')
    imgsInDir = SD.Directory()

    print("\n[INFO] Stand in the camera's view")
    SD.StoreData(cam, faceNum, imgsInDir)
    
    #keep track of face
    faceToAdd = f'face{faceNum}' #update Pickle File
    facesList.append(faceToAdd)

    #load encodings
    print("[INFO] serializing encodings...")
    faceAdded, hasEncodings = SD.StoreEncodings(f'Data/face{faceNum}/')

    print(faceAdded.isAdmin)

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
    
    return

def Recognize():
    with open('encodings.pickle', 'rb') as f:
        dictFaces = pickle.load(f)
    sfr = SimpleFacerec(dictFaces)

    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Current Video Feed:", size = (40, 1), justification = "center", font = "Helvetiva 20")],
        [sg.Image(filename = "", key = "-IMAGE-")],
        [sg.Text("Administrator Use Only:", size = (20, 1)), sg.Button("Add Administrator", size = (15, 1), pad = ((90, 0), 3), font = "Helvetica 14"), sg.Button("Add User", size = (15, 1), font = "Helvetica 14"), sg.Button("Unlock Door", size = (15, 1), font = "Helvetica 14")]
    ]

    # Create the window and show it without the plot
    window = sg.Window("OpenCV Integration", layout, location=(800, 400))

    cap = cv2.VideoCapture(0)

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED:
            break
        
        

        ret, frame = cap.read()

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, f'User: {name}' ,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

        # enterGPIO = input('Enter Command: ')
        # if enterGPIO == '1':
        #     StoreUser()

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

    window.close()


cmd = input('Enter Command: ')

while cmd != 'Quit':

    if cmd == 'Store':
        StoreUser()
        
    elif (cmd == 'Recognize'):
        Recognize()
        
    cmd = input('Enter Command: ')

# window.close()








