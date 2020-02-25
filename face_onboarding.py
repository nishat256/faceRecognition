import os
import cv2
import pickle
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

def return_files_list(path):
	"""
	  Input : path
	  Output : list of files present in provided directory(path)
	"""
	lyst_of_files = os.listdir(path)
	return lyst_of_files

def load_face_image(file_name):
	"""
	 	This function is used to load image and then resizes to shape(224,224).
	 	INPUT : path of image
	 	OUTPUT : image matrix (array)
	"""
	img = cv2.imread(file_name)
	img = img.astype('float32')
	img = cv2.resize(img,(224,224))
	return img


def get_embeddings(faces):
	"""
	   This function uses pretrained model vggface to generate embedding for image.
	   INPUT : list of images(as array)
	   OUTPUT : list of embeddings for list of images
	"""
	# convert into an array of samples
	samples = np.asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

def map_face_with_name_and_embedding():
	"""
	  This function read all images present in faces folder and asks user to provide real name of person 
	  present in that image & also asks about name of character in movie/video.It saves output as pickle file. 
	"""
	list_of_faces = return_files_list('faces/')
	images_as_array = [load_face_image('faces/'+f) for f in list_of_faces]
	embedding_of_faces =  get_embeddings(images_as_array)
	mapping = []
	for index, file_name in enumerate(list_of_faces):
		print('name of file : {}'.format(file_name))
		name = input('Enter name of person :')
		embedding = embedding_of_faces[index]
		mapping.append({'name':name ,'embedding':embedding})
	db = {}
	db['mapping'] = mapping
	print(mapping)
	dbfile = open('face_to_embedding.pickle','wb')
	pickle.dump(db,dbfile)
	dbfile.close()
	

if __name__ == "__main__":
	map_face_with_name_and_embedding()



