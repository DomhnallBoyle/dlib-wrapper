import ast
import cv2
import dlib
import json
import numpy
import os
import sys


current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'
predictor = dlib.shape_predictor(current_directory + 'shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(current_directory + 'dlib_face_recognition_resnet_model_v1.dat')

image = cv2.imread(sys.argv[1])
face_locations = list(ast.literal_eval(sys.argv[2]))

face_encodings = []

for location in face_locations:
    location = dlib.rectangle(location[0], location[1], location[2], location[3])
    shape_outline = predictor(image, location)
    encoding = face_encoder.compute_face_descriptor(image, shape_outline, 1)

    face_encodings.append(list(encoding))

print json.dumps(face_encodings)
