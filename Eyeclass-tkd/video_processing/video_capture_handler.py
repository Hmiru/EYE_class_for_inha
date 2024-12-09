import cv2

class VideoCaptureHandler:
    def __init__(self, stream_path):

        if isinstance(stream_path, str):
            self.stream_path = stream_path
        else:
            self.stream_path = 0  # 기본적으로 웹캠(0번)으로 설정

        self.cap = cv2.VideoCapture(self.stream_path)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()