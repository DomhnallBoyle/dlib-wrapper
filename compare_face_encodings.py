import ast
import functions
import json
import numpy
import sys


face_encodings = numpy.array(ast.literal_eval(sys.argv[1]))
testing = sys.argv[2]

if testing == "on":
    testing = True
else:
    testing = False

known_face_encodings, known_face_names = functions.get_known_identities(testing=testing)

# compares known faces with faces from the image provided
# returns the face names of people recognised
face_names = []

for face_encoding in face_encodings:
    matches = (numpy.linalg.norm(known_face_encodings - face_encoding, axis=1) <= 0.6).tolist()
    name = 'Unknown'

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    face_names.append(name)

print json.dumps(face_names)
