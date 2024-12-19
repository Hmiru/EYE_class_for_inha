import cv2

class VideoCaptureHandler:
    def __init__(self, stream_path):

        if isinstance(stream_path, str):
            self.stream_path = stream_path
        else:
            self.stream_path = 0  # 웹캠으로 진행하기에 프로그램 시작하기 전 cam_id 확인 필요

        self.cap = cv2.VideoCapture(self.stream_path)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()