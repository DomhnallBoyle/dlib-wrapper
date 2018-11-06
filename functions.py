import cv2
import dlib
import numpy
import os

current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'
face_detector = dlib.get_frontal_face_detector()
pose_predictor = dlib.shape_predictor(current_directory + 'shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(current_directory + 'dlib_face_recognition_resnet_model_v1.dat')
training_images_dir = current_directory + 'images/training/'


def get_known_identities(testing=False):
    # retrieves the images of known people and returns their encodings and face
    # names. Image names should be in the form of {name}.png etc and should be
    # single person images
    known_face_encodings = []
    known_face_names = []

    if testing:
        for image_name in os.listdir(training_images_dir):
            image = cv2.imread(training_images_dir + image_name)
            face_location = get_face_locations(image)
            known_face_encodings.append(get_face_encodings(image, face_location)[0])
            known_face_names.append(os.path.splitext(image_name)[0])
    else:
        # gets images from storage somewhere
        pass

    return known_face_encodings, known_face_names


def add_identity():
    # adds an image to the storage of known identies somewhere
    # should this function be here?
    pass


def get_face_encodings(image, face_locations):
    # return the 128-dimension face encoding for each face in the image.
    face_encodings = []

    for location in face_locations:
        shape_outline = pose_predictor(image, location)
        encoding = face_encoder.compute_face_descriptor(image, shape_outline, 1)

        face_encodings.append(numpy.array(encoding))

    return face_encodings
    

def get_face_locations(image):
    # returns detected face locations from an image
    return face_detector(image, 1)


def find_matches(known_face_encodings, face_encodings, known_face_names):
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

    return face_names


def convert_to_positions(rect):
    # returns positions of a rectangle as a tuple
    return rect.top(), rect.right(), rect.bottom(), rect.left()
