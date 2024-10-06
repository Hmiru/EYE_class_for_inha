class FaceBoxProvider:
    def __init__(self, face_boxes):
        self.face_boxes = face_boxes

    def get_face_boxes(self, gray_frame=None):
        return self.face_boxes