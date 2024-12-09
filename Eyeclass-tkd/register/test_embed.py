from ultralytics import YOLO
from register.model import MobileFaceNet
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

# YOLO 모델
yolo_model = YOLO("yolov11n-face.pt")

# MobileFaceNet 모델
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_net = MobileFaceNet(embedding_size=512).to(device)
state_dict = torch.load("model_mobilefacenet.pth", map_location=device)
face_net.load_state_dict(state_dict)
face_net.eval()

# 얼굴 탐지
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame)
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # 바운딩 박스 좌표
        face_img = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # MobileFaceNet 임베딩 생성
        face_pil = Image.fromarray(face_img).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = face_net(face_tensor).cpu().numpy()

    cv2.imshow("YOLO + MobileFaceNet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
