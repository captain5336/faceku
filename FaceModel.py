import dlib
import numpy as np


class FaceModel():
    """ 
    Provide functions for face detect, feature abstraction, face_recognition
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.face_encoder = dlib.face_recognition_model_v1(
            "model/dlib_face_recognition_resnet_model_v1.dat")
        self.predictor = dlib.shape_predictor(
            "model/shape_predictor_68_face_landmarks.dat")

    def face_encodings(self, face_image, num_upsample=1, num_jitters=1):
        """
        Detect faces and abstract face features
        Returns the 128D features and face rectangle for each face in the image
        """
        # Detect faces in image:
        face_locations = self.detector(face_image, num_upsample)
        # Detected landmarks:
        # raw_landmarks = [self.predictor(face_image, face_location)
        #                 for face_location in face_locations]

        rect_list = []
        raw_landmarks = []
        for face_location in face_locations:
            raw_landmarks.append(self.predictor(face_image, face_location))

            top = face_location.top()
            right = face_location.right()
            bottom = face_location.bottom()
            left = face_location.left()
            rect_list.append((top, right, bottom, left))

        print('face_locations ', rect_list)

        # Calculate the face encoding for every detected face using the detected landmarks for each one:
        list_128d = [np.array(
            self.face_encoder.compute_face_descriptor(
                face_image,
                raw_landmark_set,
                num_jitters)
        ) for raw_landmark_set in raw_landmarks
        ]

        res = {
            'list_128d': list_128d,
            'face_locations': rect_list,
        }
        return res

    def compare_faces_ordered(self, face_encodings, face_names, encoding_to_check):
        """Returns the ordered distances and names when comparing a list of face 
        encodings against a candidate feature to check """
        # calculate Euclidean distance
        euc = np.linalg.norm(face_encodings - encoding_to_check, axis=1)
        distances = list(euc)
        return zip(*sorted(zip(distances, face_names)))
