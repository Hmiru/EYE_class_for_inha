# preprocessor.py
import threading
import cv2
import numpy as np
import queue


class Preprocessor(threading.Thread):
    """
    멀티스레드 전처리 클래스
    """
    def __init__(self, input_queue, output_queue, stop_event, input_size=(640, 640)):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.input_size = input_size

    def preprocess_frame(self, frame):
        """
        단일 프레임에 대한 전처리
        """
        # 리사이즈
        resized_frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)

        # 정규화 (0~1 범위로 스케일링)
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        return normalized_frame

    def run(self):
        """
        스레드 루프: 입력 큐에서 프레임 가져와 전처리 후 출력 큐에 추가
        """
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=1)
                if frame is None:  # None이 들어오면 종료
                    break

                preprocessed_frame = self.preprocess_frame(frame)

                # 전처리된 프레임을 출력 큐에 추가
                self.output_queue.put(preprocessed_frame)
            except queue.Empty:
                continue
