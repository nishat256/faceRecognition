import cv2
import time
from facial_landmarks_detection import calculate_eye_aspect_ratio

EYE_AR_THRESH = .25
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
text = "Blink 5 times to start identification process."
COUNTER = 0
TOTAL = 0
recognize = False
cap = cv2.VideoCapture(0)
time.sleep(1.0)
while True:
	_, frame = cap.read()
	if TOTAL == 5:
		TOTAL =0
		recognize = True
		text = "Blink 5 times to start identification process."
	resp = calculate_eye_aspect_ratio(frame,recognize,text)
	recognize = False
	if resp:
		ear , frame , text= resp
	else:
		ear = 100
	# check to see if the eye aspect ratio is below the blink
	# threshold, and if so, increment the blink frame counter
	if ear < EYE_AR_THRESH:
		COUNTER += 1
	# otherwise, the eye aspect ratio is not below the blink threshold
	else:
		# if the eyes were closed for a sufficient number of
		# then increment the total number of blinks
		if COUNTER >= EYE_AR_CONSEC_FRAMES:
			TOTAL += 1
			# reset the eye frame counter
			COUNTER = 0
	# draw the total number of blinks on the frame along with
	# the computed eye aspect ratio for the frame
	cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()