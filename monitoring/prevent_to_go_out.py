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


class AbsencePrevention:
    def __init__(self, weights_path='model_mobilefacenet.pth', registered_faces_path='registered_faces.pkl',
                 device=torch.device('cpu')):
        self.device = device
        self.model = self.load_model(weights_path)
        self.face_cascade = cv2.CascadeClassifier('register/haarcascade_frontalface_default.xml')
        self.preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.registered_faces = self.load_registered_faces(registered_faces_path)
        self.last_seen = {}

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

    def absence_prevention_live(self, frame, max_absence_time=5):
        recognized_faces_info = []  # List to store recognized face info per frame
        absence_detected=False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))

        current_time = time.time()
        recognized_faces_info.clear()  # Reset for each frame

        for (x, y, w, h) in faces:
            #draw rec
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            x2, y2 = x + w, y + h  # x2, y2를 계산하여 (x, y, x2, y2) 형식으로 변환
            face_img = frame[y:y2, x:x2]
            face_pil = Image.fromarray(face_img).convert('RGB')
            face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                new_embedding = self.model(face_tensor).cpu().numpy()

            recognized_name = self.recognize_face(new_embedding)
            #cv2.putText(frame, recognized_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if recognized_name:
                self.last_seen[recognized_name] = current_time

                recognized_faces_info.append({
                    "student_id": recognized_name,
                    "bbox": (x, y, x2, y2)
                })

        # Absence detection
        for student_id, last_time in self.last_seen.items():
            time_diff = current_time - last_time
            if time_diff > max_absence_time:
                absence_detected=True
                print(f"{student_id} 이탈 감지! {time_diff:.2f}초 동안 감지되지 않음.")
                cv2.putText(frame, f"{student_id} absent detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Example output
        print("현재 프레임에서 인식된 얼굴 정보:", recognized_faces_info)


        return recognized_faces_info,absence_detected



# Usage example
if __name__ == "__main__":
    absence_prevention = AbsencePrevention(
        weights_path='../register/model_mobilefacenet.pth',
        registered_faces_path='../register/registered_faces.pkl',
        device=torch.device('cpu')
    )
    while True:
        #webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            break
        absence_prevention.absence_prevention_live(frame)
