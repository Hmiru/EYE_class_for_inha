import cv2
import os

# 이름 입력 받기 (각 조원별 폴더 생성)
name = input("Enter the name: ")
train_data_dir = f"test/{name}"
os.makedirs(train_data_dir, exist_ok=True)

# 카메라 시작
cap = cv2.VideoCapture(0)
num_images = 10  # 각 사람당 10장의 이미지 캡처
count = 0

while count < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 캡처 (다양한 각도에서 직접 촬영)
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


'''
이후, 아래 있는 디렉토리도 함께 추가하여 테스트 진행
'''