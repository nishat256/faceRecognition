import dlib
import cv2
import numpy as np
from imutils import face_utils
from face_recognition import recognize_faces_in_frame
from scipy.spatial import distance as dist

pretrained_model_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pretrained_model_path)

def detect_landmarks(image,recognize=False):
	"""
		This function is used to detect faces in image and then predict positions of facial landmarks using
		dlib library.
	"""
	# Converting the image to gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Get faces into image
	rects = detector(gray, 0)
	shape = None
	face_cordinates = None
	if rects:
		# Make the prediction and transfom it to numpy array
		x1 = rects[0].left()
		y1 = rects[0].top()
		x2 = rects[0].right()
		y2 = rects[0].bottom()
		face_cordinates = (x1,y1,x2,y2)
		shape = predictor(gray, rects[0])
		shape = face_utils.shape_to_np(shape)
	return shape,face_cordinates

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def calculate_eye_aspect_ratio(image,recognize,text):
	shape ,face_cordinates = detect_landmarks(image)
	if (isinstance(shape,np.ndarray) and not shape.shape):
		return None
	elif not isinstance(shape,np.ndarray) and not shape:
		return None
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	left_eye = shape[lStart:lEnd]
	right_eye = shape[rStart:rEnd]
	left_ear = eye_aspect_ratio(left_eye)
	right_ear = eye_aspect_ratio(right_eye)
	ear = (left_ear + right_ear) / 2.0
	# compute the convex hull for the left and right eye, then
	# visualize each of the eyes
	left_eye_hull = cv2.convexHull(left_eye)
	right_eye_hull = cv2.convexHull(right_eye)
	cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 1)
	cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 1)
	(x1,y1,x2,y2) = face_cordinates
	face_in_frame = image[y1:y2, x1:x2]
	face_in_frame = cv2.resize(face_in_frame,(224,224))
	cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,0),2)
	if recognize:
		text = recognize_faces_in_frame(face_in_frame)
	image = write_on_frame(image ,text,x1,y1)
	return (ear,image,text)

def write_on_frame(frame,text,x,y):
	"""
		This is used to put text in frame on detected face.
	"""
	font = cv2.FONT_HERSHEY_PLAIN
	org = (x,y-10)
	font_scale = .8
	color = (255,255,0)
	thickness = 1
	img = cv2.putText(frame,text,org,font,font_scale,color,thickness,cv2.LINE_AA)
	return img

if __name__ == "__main__":
	image = cv2.imread('input/vipin.jpg')
	print(calculate_eye_aspect_ratio(image))

 