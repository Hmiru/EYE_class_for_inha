from video_processing.video_capture_handler import VideoCaptureHandler
from detection.face_detector import FaceDetector
from detection.mouth_detector import MouthDetector
from detection.yawn_detector import YawnDetector
from video_processing.video_processor import VideoProcessor

if __name__ == "__main__":
    video_capture_handler = VideoCaptureHandler("video_image/2girls_yawning.mp4")
    face_detector = FaceDetector()
    mouth_detector = MouthDetector("predictor/shape_predictor_68_face_landmarks.dat")
    yawn_detector = YawnDetector()

    # face_boxes = [(100, 100, 200, 200), (300, 300, 400, 400)]
    # face_box_provider = FaceBoxProvider(face_boxes)

    video_processor = VideoProcessor(video_capture_handler, face_detector, mouth_detector, yawn_detector)
    video_processor.process()