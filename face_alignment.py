import cv2
import dlib
import json
import os
import sys
import tempfile
import zipfile


current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(current_directory + 'shape_predictor_5_face_landmarks.dat')

image_filename = sys.argv[1]
image = cv2.imread(image_filename)

detections = detector(image)

if len(detections) > 0:
    faces = dlib.full_object_detections()
    for detection in detections:
        faces.append(predictor(image, detection))

    images = dlib.get_face_chips(image, faces, size=320)

    # do something with the aligned faces here
    zip = zipfile.ZipFile(current_directory + 'face_alignment_images.zip', 'w')
    for i in range(0, len(images)):
    	filename = 'face_alignment_{}.jpg'.format(i+1)
    	cv2.imwrite(filename, images[i])
        zip.write(filename)
        os.remove(filename)
    zip.close()

    print current_directory + 'face_alignment_images.zip'
else:
    print json.dumps({'error': 'No faces detected.'})
