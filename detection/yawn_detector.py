import numpy as np

class YawnDetector:
    def __init__(self, yawn_threshold=0.7, consecutive_frames=30):
        self.yawn_threshold = yawn_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_count = 0

    def calculate_mar(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def detect_yawn(self, gray_frame, mouth_box):
        (mx, my, mw, mh) = mouth_box
        mouth = gray_frame[my:my+mh, mx:mx+mw]
        mar = self.calculate_mar(mouth)
        if mar > self.yawn_threshold:
            self.frame_count += 1
            if self.frame_count >= self.consecutive_frames:
                return True, mar
        else:
            self.frame_count = 0
        return False, 0.0