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

        cv2.line(frame, tuple(mouth[2]), tuple(mouth[10]), (255, 0, 0), 2)#blue
        cv2.line(frame, tuple(mouth[3]), tuple(mouth[9]), (0, 255, 0), 2) #green
        cv2.line(frame, tuple(mouth[0]), tuple(mouth[6]), (0, 0, 255), 2)#red



        mar = (A + B) / (2.0 * C)

        print(mar)

        return mar

    def detect_yawn(self, gray_frame, mouth_landmarks,frame):
        #(mx, my, mw, mh) = mouth_box
        #mouth = gray_frame[my:my+mh, mx:mx+mw]g
        mar = self.calculate_mar(mouth_landmarks,frame)
        ##print(mar)
        if mar > self.yawn_threshold:
            self.frame_count += 1
            if self.frame_count >= self.consecutive_frames:
                return True, mar
        else:
            self.frame_count = 0
        return False, mar