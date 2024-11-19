import cv2
import torch
from torchvision import transforms
from model import SimpleCNN
from EYE_class_for_inha.detection.face_landmark_detector import FaceLandmarkDetector
from PIL import Image  # PIL.Image 추가

# 모델 및 전처리 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("yawn_detection_model.pth"))  # 학습된 모델 경로
model.eval()

# 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# 얼굴 및 랜드마크 검출기 설정
predictor_path = "../EYE_class_for_inha/predictor/shape_predictor_68_face_landmarks.dat"  # landmarks 모델 경로
face_landmark_detector = FaceLandmarkDetector(predictor_path)

# 웹캠 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_landmark_detector.detector(gray_frame)
    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        face_box = (x1, y1, x2, y2)

        # 입 부분 추출
        mouth_landmarks = face_landmark_detector.detect_mouth_from_face_box(gray_frame, face_box)

        # 입 영역을 Bounding Box로 표시
        if mouth_landmarks is not None:
            # 입의 좌우 및 상하 경계를 계산
            x_min = min([point[0] for point in mouth_landmarks])
            y_min = min([point[1] for point in mouth_landmarks])
            x_max = max([point[0] for point in mouth_landmarks])
            y_max = max([point[1] for point in mouth_landmarks])

            # 입 영역만 추출
            mouth_region = frame[y_min:y_max, x_min:x_max]
            mouth_region_pil = Image.fromarray(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB))  # numpy -> PIL.Image
            mouth_tensor = transform(mouth_region_pil).unsqueeze(0).to(device)  # 변환 후 tensor로 변환

            # 모델을 통한 예측
            with torch.no_grad():
                output = model(mouth_tensor)
                _, predicted = torch.max(output, 1)
                label = "Yawn" if predicted.item() == 0 else "No Yawn"

            # 입 영역에 박스와 라벨 추가
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 결과 표시
    cv2.imshow("Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
