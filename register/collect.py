import cv2
import os

# 학번을 입력 받아 학번별로 얼굴 정보 저장하기 위함.
name = input("Enter the name: ")
train_data_dir = f"test/{name}"
os.makedirs(train_data_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
num_images = 10  # 각 사람당 10장의 이미지 캡처(필요에 따라 수정 가능)
count = 0

while count < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Capture Training Image', frame)

    # 'c' 키를 누르면 이미지 저장
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img_name = f"{train_data_dir}/face_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
        count += 1

    # 10장 촬영 후 자동 종료
    if count >= num_images:
        print(f"Captured {num_images} images. Exiting...")
        break

    # 'q' 키를 누르면 강제 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
