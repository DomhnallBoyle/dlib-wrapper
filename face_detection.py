import cv2
import dlib
import json
import sys


detector = dlib.get_frontal_face_detector()

image_filename = sys.argv[1]
image = cv2.imread(image_filename)

results = []

detections = detector(image, 1)

if len(detections) > 0:
    for detection in detections:
        results.append({
            'left': detection.left(),
            'top': detection.top(),
            'right': detection.right(),
            'bottom': detection.bottom()
        })

    print json.dumps(results)
else:
    print json.dumps({'error': 'No faces detected.'})
