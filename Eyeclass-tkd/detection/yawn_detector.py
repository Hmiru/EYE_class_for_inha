import numpy as np
import cv2

class YawnDetector:
    def __init__(self, yawn_threshold=0.6, consecutive_frames=7):
        self.yawn_threshold = yawn_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_count = 0

    def calculate_mar(self, mouth,frame):
        A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
        B = np.linalg.norm(mouth[3] - mouth[9])   # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55

        cv2.line(frame, tuple(mouth[2]), tuple(mouth[10]), (255, 0, 0), 2) #blue
        cv2.line(frame, tuple(mouth[3]), tuple(mouth[9]), (0, 255, 0), 2) #green
        cv2.line(frame, tuple(mouth[0]), tuple(mouth[6]), (0, 0, 255), 2) #red

        mar = (A + B) / (2.0 * C)

        return mar

