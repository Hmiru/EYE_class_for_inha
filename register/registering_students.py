# new_register.py

import os
import cv2
import torch
import pickle
import torchvision.transforms as transforms
from PIL import Image
from model import MobileFaceNet

def load_model(weights_path='model_mobilefacenet.pth', device=torch.device('cpu')):
    model = MobileFaceNet(embedding_size=512).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def extract_embedding(image_path, model, device, preprocess):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 {image_path}를 불러오지 못했습니다.")
        return None

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')     # opencv에서 제공하는 얼굴 탐지 모델
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # 얼굴 탐지 수행

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:      # 얼굴 부분만 잘라냄.
        face = image[y:y+h, x:x+w]
        face = Image.fromarray(face).convert('RGB')
        face_tensor = preprocess(face).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy()    # 임베딩 벡터 추출
        return embedding
    return None

def register_faces(directory_path, model, device, preprocess):
    registered_faces = {}
    for student_id in os.listdir(directory_path):
        student_dir = os.path.join(directory_path, student_id)
        for image_name in os.listdir(student_dir):
            image_path = os.path.join(student_dir, image_name)
            embedding = extract_embedding(image_path, model, device, preprocess)
            if embedding is not None:
                registered_faces[student_id] = embedding
                print(f"학번 {student_id}의 얼굴이 {image_name}에서 등록되었습니다.")
            else:
                print(f"{image_name}에서 얼굴을 감지하지 못했습니다.")

    with open('registered_faces.pkl', 'wb') as f:
        pickle.dump(registered_faces, f)

def main():
    preprocess = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device('cpu')
    model = load_model(device=device)

    directory_path = 'test/'
    register_faces(directory_path, model, device, preprocess)

if __name__ == "__main__":
    main()
