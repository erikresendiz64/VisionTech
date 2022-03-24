#! /usr/bin/python

# import the necessary packages
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
from time import sleep

from gpiozero import Servo, AngularServo
from gpiozero import LED
import RPi.GPIO as GPIO
from gpiozero.pins.rpigpio import RPiGPIOFactory

ledOpen = LED(20)
ledClose = LED(21)
print(cv2.__version__)
ledClose.on()
sleep(1)

RELAY = 18
factory = RPiGPIOFactory()
servo = AngularServo(RELAY, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servo.max()
sleep(1)
servo.value = None
#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "Anas"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())


# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
# vs = cv2.VideoCapture(0)
time.sleep(2.0)

cascade = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade)

# start the FPS counter
fps = FPS().start()

prevTime = 0
doorUnlock = False

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)


	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# OpenCV returns (x, y, w, h) box coordinates
	# So we reorder as we need (top, right, bottom, left)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

	# Detect the face boxes
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	# boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"], encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# Unlock Door
			# GPIO.output(RELAY, GPIO.HIGH)
			prevTime = time.time()
			doorUnlock = True
			print("Access Granted")
			servo.min()
			ledOpen.on()
			ledClose.off()
			sleep(1)
			servo.value = None

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			# if currentname != name:
			# 	currentname = name
			# 	print(currentname)
				
			# 	servo.angle = -90
			# 	time.sleep = 2
			# 	servo.angle = 90

		# update the list of names
		names.append(name)

	# Lock after 5 seconds
	if doorUnlock == True and time.time() - prevTime > 4:
		doorUnlock = False
		GPIO.output(RELAY, GPIO.LOW)
		print("Access Denied")
		servo.max()
		ledOpen.off()
		ledClose.on()
		sleep(1)
		servo.value = None


	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

	# display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
