import cv2

class VideoCaptureHandler:
    def __init__(self, stream_path):
        self.stream_path = stream_path
        self.cap = cv2.VideoCapture(self.stream_path)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()