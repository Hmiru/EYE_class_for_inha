from video_processing.video_capture_handler import VideoCaptureHandler
from detection.face_detector import FaceDetector
from detection.mouth_detector import MouthDetector
from detection.yawn_detector import YawnDetector
from detection.eye_detector import EyeDetector
from detection.eye_closed_detector import EyeClosedDetector
from detection.face_landmark_detector import FaceLandmarkDetector
from video_processing.video_processor import VideoProcessor
from monitoring.prevent_to_go_out import AbsencePrevention  # AbsencePrevention import 추가
from database.database_utils import initialize_db, load_registered_students
import torch
import threading
from database.attendance_gui import AttendanceGUI  # AttendanceGUI import 추가
import tkinter as tk  # tkinter import 추가

if __name__ == "__main__":
    #initialize_db()

    registered_students = load_registered_students("register/registered_faces.pkl")

    #video_capture_handler = VideoCaptureHandler("video_image/three.mp4")  # 파일 영상
    #video_capture_handler = VideoCaptureHandler("video_image/2girls_yawning.mp4")  # 파일 영상
    video_capture_handler = VideoCaptureHandler(0)  # 웹캠을 사용하려면 0으로 설정

    print("Handler initialized.")
    face_detector = FaceDetector()
    mouth_detector = MouthDetector("predictor/shape_predictor_68_face_landmarks.dat")
    eye_detector = EyeDetector("predictor/shape_predictor_68_face_landmarks.dat")

    yawn_detector = YawnDetector(yawn_threshold=0.7, consecutive_frames=10)
    eye_closed_detector = EyeClosedDetector(eye_threshold=0.15, consecutive_frames=20)

    absence_prevention = AbsencePrevention(
        weights_path="register/model_mobilefacenet.pth",
        registered_faces_path="register/registered_faces.pkl",
        device=torch.device("cpu")  # 필요에 따라 'cuda'로 변경 가능
    )
    video_processor = VideoProcessor(
        video_capture_handler,
        face_detector,
        mouth_detector,
        eye_detector,
        yawn_detector,
        eye_closed_detector,
        absence_prevention=absence_prevention,
        registered_students=registered_students
    )
    
    video_processor.process()
    # root = tk.Tk()
    # gui = AttendanceGUI(root)

    # def update_gui():
    #     gui.update_table()
    #     root.after(2000, update_gui)  # GUI 갱신 주기를 2초로 설정

    # def on_closing():
    #     """GUI 창 닫을 때 종료 처리를 위한 함수."""
    #     print("프로그램 종료 중...")
    #     video_processor.video_capture_handler.release()
    #     root.destroy()
    #     exit(0)

    # root.protocol("WM_DELETE_WINDOW", on_closing)

    # # Start GUI in the main thread
    # update_gui()

    # # Start video processing in a separate thread
    # video_thread = threading.Thread(target=video_processor.process, daemon=True)
    # video_thread.start()

    # # Start database fetching in a separate thread
    # db_thread = threading.Thread(target=fetch_db_data, args=(gui,), daemon=True)
    # db_thread.start()

    # # Start GUI
    # root.mainloop()