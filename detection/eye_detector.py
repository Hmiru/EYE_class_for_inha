from detection.face_landmark_detector import FaceLandmarkDetector
class EyeDetector:
    def __init__(self, predictor_path):
        self.face_landmark_detector = FaceLandmarkDetector(predictor_path)

    def detect_eyes(self, gray_frame, face_box):
        return self.face_landmark_detector.detect_eye_from_face_box(gray_frame, face_box)