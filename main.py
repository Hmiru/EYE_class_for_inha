from video_processing.video_capture_handler import VideoCaptureHandler
from detection.face_detector import FaceDetector
from detection.mouth_detector import MouthDetector
from detection.yawn_detector import YawnDetector
from video_processing.video_processor import VideoProcessor
from monitoring.prevent_to_go_out import AbsencePrevention
from database.database_utils import initialize_db, load_registered_students
import torch
import threading
from database.attendance_gui import AttendanceGUI
import tkinter as tk

if __name__ == "__main__":
    #DB 초기화
    initialize_db()

    registered_students = load_registered_students("register/registered_faces.pkl")

    video_capture_handler = VideoCaptureHandler(0)  # 시작하기 전 cam_id 확인 후 실행

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
        device=torch.device("cpu")
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

    update_gui()

    video_thread = threading.Thread(target=video_processor.process, daemon=False)
    video_thread.start()

    root.mainloop()


