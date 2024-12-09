class FaceBoxProvider:
    def __init__(self):
        self.faces_info = []

    def update_faces(self, faces_info):
        """Update with a new list of recognized faces info."""
        self.faces_info = faces_info

    def get_face_boxes(self):
        """Return a list of bounding boxes and ids for recognized faces."""
        return [(info["student_id"], info["bbox"]) for info in self.faces_info]
