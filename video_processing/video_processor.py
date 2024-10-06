import cv2

class VideoProcessor:
    def __init__(self, video_capture_handler, face_detector, mouth_detector, yawn_detector, face_box_provider=None):
        self.video_capture_handler = video_capture_handler
        self.face_detector = face_detector
        self.mouth_detector = mouth_detector
        self.yawn_detector = yawn_detector
        self.face_box_provider = face_box_provider

    def process(self):
        while True:
            frame = self.video_capture_handler.get_frame()
            if frame is None:
                break

            frame = cv2.resize(frame, (800, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.face_box_provider:
                faces_bounding_boxes = self.face_box_provider.get_face_boxes()
            else:
                faces_bounding_boxes = self.face_detector.detect_faces(gray)

            for rect in faces_bounding_boxes:
                x1, y1, x2, y2 = rect
                mouth_box = self.mouth_detector.detect_mouth(gray, rect)

                if mouth_box:
                    (mx, my, mw, mh) = mouth_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

                    yawn_detected, mar = self.yawn_detector.detect_yawn(gray, mouth_box)
                    if yawn_detected:
                        cv2.putText(frame, "Yawn Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture_handler.release()