import cv2
import dlib
import json
import os
import sys


current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

detector = dlib.cnn_face_detection_model_v1(current_directory + 'mmod_human_face_detector.dat')

image_filename = sys.argv[1]
image = cv2.imread(image_filename)

results = []

detections = detector(image, 1)

if len(detections) > 0:
    for detection in detections:
        results.append({
            'rect': {
                'left': detection.rect.left(),
                'top': detection.rect.top(),
                'right': detection.rect.right(),
                'bottom': detection.rect.bottom()
            },
            'confidence': round(detection.confidence, 2)
        })

    print json.dumps(results)
else:
    print json.dumps({'error': 'No faces detected.'})
