import cv2
import dlib
import json
import os
import sys


current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(current_directory + 'shape_predictor_68_face_landmarks.dat')

image_filename = sys.argv[1]
image = cv2.imread(image_filename)

components = {
    'face_contour': [0, 16],
    'left_eyebrow': [17, 21],
    'right_eyebrow': [22, 26],
    'nose': [27, 35],
    'left_eye': [36, 41],
    'right_eye': [42, 47],
    'mouth': [48, 67]
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
