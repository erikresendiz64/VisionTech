import cv2 as cv
import tensorflow
import matplotlib
import face_recognition

cam = cv.VideoCapture(0)

cv.namedWindow("Python Cam Screenshot")

name = 'Anas'
img_counter = 0

while True:
    ret, frame = cam.read()

    # print(ret)
    # print(frame)

    if not ret:
        print("Failed to grab frame")
        break

    cv.imshow("Press Space to take a photo", frame)

    k = cv.waitKey(1)

    if k % 256 == 27:
        print("Esc hit, closing app")
        break
    elif k % 256 == 32:
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv.imwrite(img_name, frame)
        print("Screenshot taken")
        img_counter += 1

for i in range(img_counter):
    image = face_recognition.load_image_file("dataset/"+ name +"/image_{}.jpg".format(i))
    face_locations = face_recognition.face_locations(image)
    print(face_locations)

cam.release()
cam.destroyAllWindows()