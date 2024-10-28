from video_processing.video_capture_handler import VideoCaptureHandler
from detection.face_detector import FaceDetector
from detection.mouth_detector import MouthDetector
from detection.yawn_detector import YawnDetector
from video_processing.video_processor import VideoProcessor
from detection.yawning_counter import yawningCounter

if __name__ == "__main__":
    # video_capture_handler = VideoCaptureHandler("video_image/2girls_yawning.mp4")  # 파일 영상
    video_capture_handler = VideoCaptureHandler(0)  # 웹캠을 사용하려면 0으로 설정
    print("Handler initialized.")
    face_detector = FaceDetector()
    mouth_detector = MouthDetector("predictor/shape_predictor_68_face_landmarks.dat")
    yawn_detector = YawnDetector(yawn_threshold=0.75, consecutive_frames=7)

    yawn_counter = yawningCounter()

    video_processor = VideoProcessor(
        video_capture_handler,
        face_detector,
        mouth_detector,
        yawn_detector,
        yawn_counter
    )

    video_processor.process()
