import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from glob import glob
from PIL import Image


class YawnDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []

        # yawn과 no_yawn 폴더의 이미지를 각각 불러옵니다.
        for label, sub_dir in enumerate(['yawn', 'no_yawn']):
            images = glob(os.path.join(root_dir, sub_dir, "*.jpg"))
            for img_path in images:
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")  # 흑백 이미지로 로드

        if self.transform:
            image = self.transform(image)

        return image, label
