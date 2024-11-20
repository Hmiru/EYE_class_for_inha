import numpy as np
import cv2
from detection.face_landmark_detector import FaceLandmarkDetector
class EyeClosedDetector:
    def __init__(self, eye_threshold=0.2, consecutive_frames=7):
        self.eye_threshold = eye_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_count = 0

    def calculate_ear(self, eye, frame):
        # 눈의 EAR 계산을 위해 눈의 랜드마크 좌표를 받아 처리
        A = np.linalg.norm(eye[1] - eye[5])  # 2, 6
        B = np.linalg.norm(eye[2] - eye[4])  # 3, 5
        C = np.linalg.norm(eye[0] - eye[3])  # 1, 4

        # EAR 계산
        ear = (A + B) / (2.0 * C)

        # 디버그용으로 눈에 선을 그려줌
        cv2.line(frame, tuple(eye[1]), tuple(eye[5]), (255, 0, 0), 2)  # blue
        cv2.line(frame, tuple(eye[2]), tuple(eye[4]), (0, 255, 0), 2)  # green
        cv2.line(frame, tuple(eye[0]), tuple(eye[3]), (0, 0, 255), 2)  # red

        return ear

