import cv2
import functions
import json
import sys

image_filename = sys.argv[1]
image = cv2.imread(image_filename)

known_face_encodings, known_face_names = functions.get_known_identities(testing=True)
face_locations = functions.get_face_locations(image=image)

result = {}

if len(face_locations) > 0:
    face_encodings = functions.get_face_encodings(image=image, face_locations=face_locations)
    print face_encodings
    face_names = functions.find_matches(known_face_encodings=known_face_encodings,
                                        face_encodings=face_encodings,
                                        known_face_names=known_face_names)

    print json.dumps({'faces': face_names})
else:
    json.dumps({'error': 'No faces detected.'})
