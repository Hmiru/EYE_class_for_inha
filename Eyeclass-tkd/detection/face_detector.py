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
        # YOLO 모델 로드
        self.detector = YOLO(yolo_weights)

    def detect_faces(self, frame):
        # YOLO를 사용해 얼굴 검출
        results = self.detector(frame)

        # YOLO 결과에서 바운딩 박스를 추출
        faces = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # YOLO 바운딩 박스 좌표
            faces.append((x1, y1, x2, y2))
        return faces
