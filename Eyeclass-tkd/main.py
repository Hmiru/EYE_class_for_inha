from video_processing.video_capture_handler import VideoCaptureHandler
from detection.face_detector import FaceDetector
from detection.mouth_detector import MouthDetector
from detection.yawn_detector import YawnDetector
from detection.face_landmark_detector import FaceLandmarkDetector
from video_processing.video_processor import VideoProcessor
from monitoring.prevent_to_go_out import AbsencePrevention  # AbsencePrevention import 추가
from database.database_utils import initialize_db, load_registered_students
import torch
import threading
import cv2
from database.attendance_gui import AttendanceGUI
import tkinter as tk

if __name__ == "__main__":
    initialize_db()

    registered_students = load_registered_students("register/registered_faces.pkl")

    video_capture_handler = VideoCaptureHandler(0)  # 웹캠을 사용하려면 0으로 설정
    video_capture_handler.cap.set(cv2.CAP_PROP_FPS, 30)

    print("Handler initialized.")
    face_detector = FaceDetector(
        yolo_weights="register/yolov11n-face.pt"
    )
    mouth_detector = MouthDetector("predictor/shape_predictor_68_face_landmarks.dat")
    yawn_detector = YawnDetector(yawn_threshold=0.5, consecutive_frames=7)

    absence_prevention = AbsencePrevention(
        weights_path="register/model_mobilefacenet.pth",
        registered_faces_path="register/registered_faces.pkl",
        yolo_weights="register/yolov11n-face.pt",
        device=torch.device("cpu")  # 필요에 따라 'cuda'로 변경 가능
    )

    video_processor = VideoProcessor(
        video_capture_handler,
        face_detector,
        mouth_detector,
        yawn_detector,
        absence_prevention=absence_prevention,
        registered_students=registered_students,
        db_path="attendance.db",
    )

    root = tk.Tk()
    gui = AttendanceGUI(root)


    def update_gui():
        gui.update_table()
        root.after(1000, update_gui)


    # Start GUI in the main thread
    update_gui()

    # Start video processing in a separate thread
    video_thread = threading.Thread(target=video_processor.process, daemon=False)
    video_thread.start()

    # Start GUI
    root.mainloop()


