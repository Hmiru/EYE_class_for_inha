#face_detector.py
import dlib

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, gray_frame):
        faces = self.detector(gray_frame, 1)
        return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]