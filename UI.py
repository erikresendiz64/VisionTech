import PySimpleGUI as sg
import cv2
import numpy as np

def main():
    sg.theme("LightGreen")

    # Define the window layout
    # layout = [
    #     [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
    #     [sg.Image(filename="", key="-IMAGE-")],
    #     [sg.Button("Exit", size=(10, 1))],
    # ]

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
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()


        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()