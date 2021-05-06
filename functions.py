from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import Normalizer

def extract_face(path, detector=MTCNN(), required_size=(160,160), save_faces=True):
  # load image from file
  image = Image.open(path)

  # convert to rgb
  image = image.convert('RGB')

  # covert to array
  pixels = np.asarray(image)

  #detect faces in the image
  results = detector.detect_faces(pixels)

  # extract the bounding box from the first face
  x1, y1, width, height = results[0]['box']

  # bug fix
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height

  # extract the face
  face = pixels[y1:y2, x1:x2]

  # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize(required_size)

  face_array = np.asarray(image)
  return face_array


def get_embedding(model, face_pixels):
  # scale pixel values
  face_pixels = face_pixels.astype('float32')

  # normalization
  mean, std = face_pixels.mean(), face_pixels.std()
  face_pixels = (face_pixels - mean) / std
  
  # transform face into one sample
  samples = np.expand_dims(face_pixels, axis=0)

  # make predictions to get embeddings
  yhat = model.predict(samples)

  return yhat[0]

def predict_using_distance(faces_embeddings, labels, face_to_predict_embeddings):
  # normalize input vector
  in_encoder = Normalizer(norm='l2')
  faces_embeddings = in_encoder.transform(faces_embeddings)
  face_to_predict_embeddings = in_encoder.transform(face_to_predict_embeddings)

  # use euclidean distance
  # the distance gives how similar the faces are
  face_distance = np.linalg.norm(faces_embeddings - face_to_predict_embeddings, axis=1)

  name = 'Unknown'

  # put threshold for the distance to know if the person is found or not
  threshold = 0.7

  # list of matching people
  matching = []
  for i in range(len(face_distance)):
    if(face_distance[i] < threshold):
      matching.append([face_distance[i], labels[i]])
  
  min_label = 'Unknown'
  min_dist = 213124
  for i in range(len(matching)):
    if(matching[i][0] < min_dist):
      min_dist = matching[i][0]
      min_label = matching[i][1]
    
  return min_label