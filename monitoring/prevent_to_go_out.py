import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from register.model import MobileFaceNet
from scipy.spatial.distance import cosine
import pickle
import time
from ultralytics import YOLO  # YOLOv8 및 YOLOv11 지원

class AbsencePrevention:
    def __init__(self, weights_path='model_mobilefacenet.pth', registered_faces_path='registered_faces.pkl',
                 yolo_weights='yolov11n-face.pt', device=torch.device('cpu')):
        self.device = device
        self.model = self.load_model(weights_path)
        self.preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.registered_faces = self.load_registered_faces(registered_faces_path)
        self.last_seen = {}

        # Load YOLO model for face detection
        self.yolo_model = YOLO(yolo_weights)

    def load_model(self, weights_path):
        model = MobileFaceNet(embedding_size=512).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def load_registered_faces(self, registered_faces_path):
        with open(registered_faces_path, 'rb') as f:
            registered_faces = pickle.load(f)
        print("등록된 얼굴 임베딩 정보를 불러왔습니다.")
        return registered_faces

    def recognize_face(self, new_embedding):
        min_distance = float('inf')
        recognized_name = None
        new_embedding = new_embedding.flatten()
        for student_id, reg_embeddings in self.registered_faces.items():
            reg_embeddings = reg_embeddings.flatten()
            distance = cosine(reg_embeddings, new_embedding)
            if distance < min_distance and distance < 0.5:
                min_distance = distance
                recognized_name = student_id
        return recognized_name

    # def absence_prevention_live(self, frame, max_absence_time=5):
    #     recognized_faces_info = []
    #     absence_detected = False
    #
    #     fixed_size = (480, 480)
    #     frame_resized = cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
    #
    #     #frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_AREA)
    #     # YOLO face detection
    #     results = self.yolo_model(frame_resized, imgsz=480)
    #
    #     # Process detected faces
    #     current_time = time.time()
    #     for result in results[0].boxes:
    #         x1, y1, x2, y2 = map(int, result.xyxy[0])  # YOLO bounding box
    #         face_img = frame_resized[y1:y2, x1:x2]
    #         face_pil = Image.fromarray(face_img).convert('RGB')
    #         face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)
    #
    #         with torch.no_grad():
    #             new_embedding = self.model(face_tensor).cpu().numpy()
    #
    #         recognized_name = self.recognize_face(new_embedding)
    #         if recognized_name:
    #             self.last_seen[recognized_name] = current_time
    #
    #             recognized_faces_info.append({
    #                 "student_id": recognized_name,
    #                 "bbox": (x1, y1, x2, y2)
    #             })
    #
    #     # Absence detection
    #     for student_id, last_time in self.last_seen.items():
    #         time_diff = current_time - last_time
    #         if time_diff > max_absence_time:
    #             absence_detected = True
    #             print(f"{student_id} 이탈 감지! {time_diff:.2f}초 동안 감지되지 않음.")
    #
    #     print("현재 프레임에서 인식된 얼굴 정보:", recognized_faces_info)
    #     return recognized_faces_info, absence_detected

    def absence_prevention_live(self, frame, max_absence_time=5):
        recognized_faces_info = []
        absence_detected = False

        # YOLO 입력 크기로 프레임 조정
        fixed_size = (480, 480)
        original_height, original_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)

        # YOLO 얼굴 탐지 수행
        results = self.yolo_model(frame_resized, imgsz=480)

        # 현재 시간 기록
        current_time = time.time()

        # YOLO 탐지된 얼굴 바운딩 박스 처리
        if results[0].boxes is not None:
            for result in results[0].boxes:
                # 리사이즈된 이미지 기준의 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, result.xyxy[0])

                # 원본 이미지 크기로 좌표 변환
                x1 = int(x1 * original_width / fixed_size[0])
                x2 = int(x2 * original_width / fixed_size[0])
                y1 = int(y1 * original_height / fixed_size[1])
                y2 = int(y2 * original_height / fixed_size[1])

                # 얼굴 영역 추출 및 임베딩 계산
                face_img = frame[y1:y2, x1:x2]
                face_pil = Image.fromarray(face_img).convert('RGB')
                face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    new_embedding = self.model(face_tensor).cpu().numpy()

                # 얼굴 인식
                recognized_name = self.recognize_face(new_embedding)
                if recognized_name:
                    # 마지막 감지 시간 업데이트
                    self.last_seen[recognized_name] = current_time

                    # 인식된 얼굴 정보 추가
                    recognized_faces_info.append({
                        "student_id": recognized_name,
                        "bbox": (x1, y1, x2, y2)
                    })

        # 이탈 감지 로직
        for student_id, last_time in self.last_seen.items():
            time_diff = current_time - last_time
            if time_diff > max_absence_time:
                absence_detected = True
                print(f"{student_id} 이탈 감지! {time_diff:.2f}초 동안 감지되지 않음.")

        print("현재 프레임에서 인식된 얼굴 정보:", recognized_faces_info)

        # YOLO 탐지 결과와 얼굴 정보를 반환
        return recognized_faces_info, results[0]


# Usage example
if __name__ == "__main__":
    absence_prevention = AbsencePrevention(
        weights_path='../register/model_mobilefacenet.pth',
        registered_faces_path='../register/registered_faces.pkl',
        yolo_weights='../register/yolov11n-face.pt',
        device=torch.device('cpu')
    )

    frame_count = 0
    cap = cv2.VideoCapture(0)
    prev_time = time.time()  # 이전 프레임 시작 시간
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        # YOLO 추론: 매 3번째 프레임에서만 실행
        if frame_count % 5 == 0:
            results = absence_prevention.yolo_model(frame, imgsz=320)  # YOLO 입력 해상도 축소
            last_results = results
        else:
            results = last_results  # 이전 YOLO 결과 재사용

        frame_count += 1

        # Measure time for each frame
        start_time = time.time()
        absence_prevention.absence_prevention_live(frame)
        end_time = time.time()

        # Calculate FPS
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        # Display frame with FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Absence Prevention", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
