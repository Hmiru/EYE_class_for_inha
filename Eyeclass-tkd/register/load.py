import torch

try:
    model = torch.load("yolov11n-face.pt", map_location="cpu")
    print(model.keys())  # 모델 메타데이터 확인
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
