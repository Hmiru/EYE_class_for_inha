# import dlib
#
# class FaceDetector:
#     def __init__(self):
#         self.detector = dlib.get_frontal_face_detector()
#
#     def detect_faces(self, gray_frame):
#         faces = self.detector(gray_frame, 1)
#         return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]


from ultralytics import YOLO


class FaceDetector:
    def __init__(self, yolo_weights='yolov11n-face.pt'):
        self.detector = YOLO(yolo_weights)

    def detect_faces(self, frame):
        # 단일 프레임 처리
        results = self.detector(frame)
        faces = [
            (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            for box in results[0].boxes
        ]
        return faces

    def detect_faces_batch(self, batch_frames):
        # 배치 처리
        results = self.detector(batch_frames)
        batch_faces = []
        for result in results:
            faces = [
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                for box in result.boxes
            ]
            batch_faces.append(faces)
        return batch_faces
