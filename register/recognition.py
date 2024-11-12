# new_recog.py

import cv2
import torch
import pickle
import torchvision.transforms as transforms
from PIL import Image
from model import MobileFaceNet
from scipy.spatial.distance import cosine

def load_model(weights_path='model_mobilefacenet.pth', device=torch.device('cpu')):
    model = MobileFaceNet(embedding_size=512).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def recognize_face(registered_faces, new_embedding):
    min_distance = float('inf')
    recognized_name = None
    for student_id, reg_embedding in registered_faces.items():
        distance = cosine(reg_embedding.flatten(), new_embedding.flatten())     # 코사인 유사도 활용
        if distance < min_distance and distance < 0.5:
            min_distance = distance
            recognized_name = student_id
    return recognized_name

def face_recognition_live(model, device, preprocess):
    with open('registered_faces.pkl', 'rb') as f:
        registered_faces = pickle.load(f)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    total_faces = 0
    correct_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_img).convert('RGB')
            face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                new_embedding = model(face_tensor).cpu().numpy()        # 새로운 얼굴에 대한 임베딩 벡터 추출

            recognized_name = recognize_face(registered_faces, new_embedding)
            total_faces += 1
            if recognized_name:
                correct_faces += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Recognized: {recognized_name}", (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face registration required", (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        accuracy = (correct_faces / total_faces) * 100 if total_faces > 0 else 0

        cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    preprocess = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device('cpu')
    model = load_model(device=device)

    face_recognition_live(model, device, preprocess)

if __name__ == "__main__":
    main()
