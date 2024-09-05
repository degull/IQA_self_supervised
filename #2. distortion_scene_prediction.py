import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import random
from transformers import ViTModel

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# 왜곡 유형 사전
distortion_label_to_idx = {
    'jpeg2000 compression': 0,
    'jpeg compression': 1,
    'white noise': 2,
    'gaussian blur': 3,
    'fastfading': 4,
    'fnoise': 5,
    'contrast': 6,
    'lens': 7,
    'motion': 8,
    'diffusion': 9,
    'shifting': 10,
    'color quantization': 11,
    'oversaturation': 12,
    'desaturation': 13,
    'white with color': 14,
    'impulse': 15,
    'multiplicative': 16,
    'white noise with denoise': 17,
    'brighten': 18,
    'darken': 19,
    'shifting the mean': 20,
    'jitter': 21,
    'noneccentricity patch': 22,
    'pixelate': 23,
    'quantization': 24,
    'color blocking': 25,
    'sharpness': 26,
    'realistic blur': 27,
    'realistic noise': 28,
    'underexposure': 29,
    'overexposure': 30,
    'realistic contrast change': 31,
    'other realistic': 32
}

# 장면 유형 사전
scene_label_to_idx = {
    'animal': 0,
    'cityscape': 1,
    'human': 2,
    'indoor': 3,
    'landscape': 4,
    'night': 5,
    'plant': 6,
    'still_life': 7,
    'others': 8
}

# 품질 범주 사전
quality_label_to_idx = {
    'bad': 0,
    'poor': 1,
    'fair': 2,
    'good': 3,
    'perfect': 4
}


# 데이터셋 준비
class IQADataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(txt_file, sep="\t", header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        distortion_label = self.img_labels.iloc[idx, 2]
        scene_label = self.img_labels.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, distortion_label, scene_label

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 경로
img_dir = 'C:/Users/IIPL02/Desktop/LIQE_nonIndex/IQA_Database/kadid10k'
txt_file = 'C:/Users/IIPL02/Desktop/LIQE_nonIndex/IQA_Database/kadid10k/splits2/kadid10k_val_clip.txt'

# 데이터셋 로드
dataset = IQADataset(txt_file=txt_file, img_dir=img_dir, transform=transform)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=seed)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# 모델 정의
class ViTDistortionScenePredictor(nn.Module):
    def __init__(self):
        super(ViTDistortionScenePredictor, self).__init__()
        # ViT 모델 로드 (특징 추출용)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # 왜곡 유형 예측을 위한 분류기
        self.distortion_classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, len(distortion_label_to_idx))  # 왜곡 유형 수
        )

        # 장면 정보 예측을 위한 분류기
        self.scene_classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, len(scene_label_to_idx))  # 장면 유형 수
        )

    def forward(self, x):
        vit_outputs = self.vit(pixel_values=x)
        encoded = vit_outputs.last_hidden_state[:, 0]  # CLS 토큰 사용

        distortion_pred = self.distortion_classifier(encoded)
        scene_pred = self.scene_classifier(encoded)

        return distortion_pred, scene_pred

# 모델 초기화
model = ViTDistortionScenePredictor().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# 학습 루프 정의
num_epochs = 36
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, distortion_labels, scene_labels in train_loader:
        images = images.to(device)

        # 레이블을 인덱스로 변환 후 GPU로 전송
        distortion_labels = torch.tensor([distortion_label_to_idx[label] for label in distortion_labels]).to(device)
        scene_labels = torch.tensor([scene_label_to_idx[label] for label in scene_labels]).to(device)

        # Forward Pass
        distortion_pred, scene_pred = model(images)

        # 손실 계산
        loss_distortion = criterion(distortion_pred, distortion_labels)
        loss_scene = criterion(scene_pred, scene_labels)
        loss = loss_distortion + loss_scene

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct_distortion = 0
    correct_scene = 0
    total = 0
    with torch.no_grad():
        for images, distortion_labels, scene_labels in val_loader:
            images = images.to(device)
            distortion_labels = torch.tensor([distortion_label_to_idx[label] for label in distortion_labels]).to(device)
            scene_labels = torch.tensor([scene_label_to_idx[label] for label in scene_labels]).to(device)

            # Forward Pass
            distortion_pred, scene_pred = model(images)

            # 손실 계산
            loss_distortion = criterion(distortion_pred, distortion_labels)
            loss_scene = criterion(scene_pred, scene_labels)
            loss = loss_distortion + loss_scene
            val_loss += loss.item() * images.size(0)

            # 정확도 계산
            _, predicted_distortion = torch.max(distortion_pred, 1)
            _, predicted_scene = torch.max(scene_pred, 1)
            total += images.size(0)
            correct_distortion += (predicted_distortion == distortion_labels).sum().item()
            correct_scene += (predicted_scene == scene_labels).sum().item()

    val_loss /= len(val_loader.dataset)
    distortion_accuracy = 100 * correct_distortion / total
    scene_accuracy = 100 * correct_scene / total
    print(f"Validation Loss: {val_loss:.4f}, Distortion Accuracy: {distortion_accuracy:.2f}%, Scene Accuracy: {scene_accuracy:.2f}%")

torch.save(model.state_dict(), 'fin_distortion_scene_prediction_model.pth')


"""

Epoch 1/36, Loss: 4.6520
Validation Loss: 3.8580, Distortion Accuracy: 9.00%, Scene Accuracy: 99.00%
Epoch 2/36, Loss: 3.5771
Validation Loss: 3.4549, Distortion Accuracy: 16.00%, Scene Accuracy: 99.50%
Epoch 3/36, Loss: 3.2769
Validation Loss: 3.2860, Distortion Accuracy: 24.00%, Scene Accuracy: 99.50%
Epoch 4/36, Loss: 3.0526
Validation Loss: 3.1147, Distortion Accuracy: 26.50%, Scene Accuracy: 99.50%
Epoch 5/36, Loss: 2.8043
Validation Loss: 2.9208, Distortion Accuracy: 35.50%, Scene Accuracy: 99.50%
Epoch 6/36, Loss: 2.5207
Validation Loss: 2.6718, Distortion Accuracy: 40.50%, Scene Accuracy: 99.50%
Epoch 7/36, Loss: 2.2281
Validation Loss: 2.4156, Distortion Accuracy: 48.50%, Scene Accuracy: 99.50%
Epoch 8/36, Loss: 1.9603
Validation Loss: 2.1763, Distortion Accuracy: 52.50%, Scene Accuracy: 99.50%
Epoch 9/36, Loss: 1.6987
Validation Loss: 1.9435, Distortion Accuracy: 51.00%, Scene Accuracy: 99.00%
Epoch 10/36, Loss: 1.4656
Validation Loss: 1.7406, Distortion Accuracy: 53.50%, Scene Accuracy: 99.00%
Epoch 11/36, Loss: 1.2872
Validation Loss: 1.6065, Distortion Accuracy: 55.50%, Scene Accuracy: 99.50%
Epoch 12/36, Loss: 1.1173
Validation Loss: 1.4737, Distortion Accuracy: 58.50%, Scene Accuracy: 99.00%
Epoch 13/36, Loss: 0.9778
Validation Loss: 1.3914, Distortion Accuracy: 58.00%, Scene Accuracy: 99.00%
Epoch 14/36, Loss: 0.8959
Validation Loss: 1.3196, Distortion Accuracy: 58.00%, Scene Accuracy: 99.00%
Epoch 15/36, Loss: 0.8085
Validation Loss: 1.2320, Distortion Accuracy: 61.00%, Scene Accuracy: 99.50%
Epoch 16/36, Loss: 0.7436
Validation Loss: 1.1741, Distortion Accuracy: 59.00%, Scene Accuracy: 99.00%
Epoch 17/36, Loss: 0.6242
Validation Loss: 1.1057, Distortion Accuracy: 64.50%, Scene Accuracy: 99.50%
Epoch 18/36, Loss: 0.5700
Validation Loss: 1.0789, Distortion Accuracy: 65.00%, Scene Accuracy: 99.00%
Epoch 19/36, Loss: 0.5135
Validation Loss: 1.0311, Distortion Accuracy: 64.50%, Scene Accuracy: 99.50%
Epoch 20/36, Loss: 0.4670
Validation Loss: 1.0150, Distortion Accuracy: 67.00%, Scene Accuracy: 99.50%
Epoch 21/36, Loss: 0.4267
Validation Loss: 1.0232, Distortion Accuracy: 67.00%, Scene Accuracy: 99.50%
Epoch 22/36, Loss: 0.3973
Validation Loss: 0.9966, Distortion Accuracy: 67.50%, Scene Accuracy: 99.50%
Epoch 23/36, Loss: 0.3981
Validation Loss: 1.0566, Distortion Accuracy: 66.00%, Scene Accuracy: 99.50%
Epoch 24/36, Loss: 0.3731
Validation Loss: 0.9459, Distortion Accuracy: 69.50%, Scene Accuracy: 99.50%
Epoch 25/36, Loss: 0.3109
Validation Loss: 0.9539, Distortion Accuracy: 71.00%, Scene Accuracy: 99.50%
Epoch 26/36, Loss: 0.2870
Validation Loss: 0.9481, Distortion Accuracy: 72.00%, Scene Accuracy: 99.50%
Epoch 27/36, Loss: 0.2613
Validation Loss: 0.9464, Distortion Accuracy: 69.50%, Scene Accuracy: 99.50%
Epoch 28/36, Loss: 0.2637
Validation Loss: 1.0120, Distortion Accuracy: 66.50%, Scene Accuracy: 99.50%
Epoch 29/36, Loss: 0.2781
Validation Loss: 0.9511, Distortion Accuracy: 72.50%, Scene Accuracy: 99.50%
Epoch 30/36, Loss: 0.2671
Validation Loss: 0.8781, Distortion Accuracy: 72.00%, Scene Accuracy: 99.50%
Epoch 31/36, Loss: 0.2917
Validation Loss: 1.0013, Distortion Accuracy: 68.00%, Scene Accuracy: 99.50%
Epoch 32/36, Loss: 0.2664
Validation Loss: 1.0253, Distortion Accuracy: 70.50%, Scene Accuracy: 99.50%
Epoch 33/36, Loss: 0.2476
Validation Loss: 0.9530, Distortion Accuracy: 68.50%, Scene Accuracy: 99.50%
Epoch 34/36, Loss: 0.2316
Validation Loss: 0.9292, Distortion Accuracy: 72.50%, Scene Accuracy: 99.50%
Epoch 35/36, Loss: 0.2195
Validation Loss: 0.9959, Distortion Accuracy: 68.50%, Scene Accuracy: 99.50%
Epoch 36/36, Loss: 0.2084
Validation Loss: 0.8586, Distortion Accuracy: 76.00%, Scene Accuracy: 99.50%
"""
