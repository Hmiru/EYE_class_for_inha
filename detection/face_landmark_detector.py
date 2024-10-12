import dlib
from imutils import face_utils
import cv2

class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_mouth_from_face_box(self, gray_frame, face_box):
        (x1, y1, x2, y2) = face_box
        rect = dlib.rectangle(x1, y1, x2, y2)
        shape = self.predictor(gray_frame, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[48:68]
        #(mx, my, mw, mh) = cv2.boundingRect(mouth)
        return mouth