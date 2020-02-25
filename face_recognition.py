import pickle
import cv2
from face_onboarding import get_embeddings
from scipy.spatial.distance import cosine

def load_pickled_data():
	"""
	  This function loads pickle object that contains embedding of characters with their details like realname etc.
	"""
	file_name = 'face_to_embedding.pickle'
	with open(file_name,'rb') as file_object:
		db = pickle.load(file_object)
	return db['mapping']

def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	print(score)
	if score <= thresh:
		return True
	else:
		return False

def recognize_faces_in_frame(detected_face):
	"""
	  This function detects and recognizes faces present in images.
	"""
	db = load_pickled_data()
	embeddings_of_detected_face = get_embeddings([detected_face])[0]
	resp_mapping = check_embedding_present_in_db(db,embeddings_of_detected_face)
	if resp_mapping:
		name = resp_mapping['name']
		return 'Identity : '+ name
	return 'Identity : Unknown'
		

def check_embedding_present_in_db(db,known_embedding):
	"""
		This is used to find details of detected faces by comparing their embeddings with embeddings present in db.
	"""
	for record in db:
		candidate_embedding = record['embedding']
		resp_match = is_match(known_embedding,candidate_embedding)
		if resp_match:
			return record
	return None
