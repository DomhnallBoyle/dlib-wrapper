import cv2
import dlib
import json
import os
import sys


current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(current_directory + 'shape_predictor_5_face_landmarks.dat')

image_filename = sys.argv[1]
image = cv2.imread(image_filename)

components = {
    'right_eye': [0, 2],
    'left_eye': [2, 4],
    'nose': [4, 5]
}

results = []

detections = detector(image, 1)

if len(detections) > 0:
    for detection in detections:
        shape = predictor(image, detection)
        face = {}
        for part, position in components.iteritems():
            data_points = []
            for point in range(position[0], position[1]):
                x, y = int(shape.part(point).x), int(shape.part(point).y)
                data_points.append(
                    (x, y)
                )
            face[part] = data_points

        results.append(
            face
        )

    print json.dumps(results)
else:
    print json.dumps({'error': 'No faces detected.'})
