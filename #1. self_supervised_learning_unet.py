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
from pytorch_msssim import SSIM  # SSIM 손실 함수 사용

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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
        
        if self.transform:
            image = self.transform(image)
        
        return image

# 이미지 전처리 및 데이터 증대 정의
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
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

# ViT 기반 인코더와 U-Net 디코더를 결합한 Autoencoder 모델 정의
class ViTUAutoencoder(nn.Module):
    def __init__(self):
        super(ViTUAutoencoder, self).__init__()
        # ViT 모델 로드 (인코더 역할)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # U-Net 구조의 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 3, kernel_size=1),  # RGB 채널 복원
            nn.Sigmoid()  # [0, 1] 범위로 값을 제한
        )

    def forward(self, x):
        # ViT 인코더로 특징 추출 (batch_size, 196, 768)
        vit_outputs = self.vit(pixel_values=x)
        encoded = vit_outputs.last_hidden_state[:, 1:, :]  # CLS 토큰 제외
        encoded = encoded.permute(0, 2, 1).contiguous()  # (batch_size, 768, 196)
        encoded = encoded.view(encoded.size(0), 768, 14, 14)  # (batch_size, 768, 14, 14)로 reshape

        # U-Net 디코더 통과
        decoded = self.decoder(encoded)
        return decoded

model = ViTUAutoencoder().to(device)

# 손실 함수 및 옵티마이저 정의
criterion_mse = nn.MSELoss()  # 복원 손실 함수
criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
optimizer = optim.AdamW(model.parameters(), lr=0.00005)  # 학습률 조정
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 학습 루프 정의
num_epochs = 100
best_val_loss = float('inf')
early_stop_patience = 10
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        
        # Forward Pass
        outputs = model(images)
        
        # 복합 손실 계산
        loss_mse = criterion_mse(outputs, images)
        loss_ssim = 1 - criterion_ssim(outputs, images)  # SSIM을 사용한 추가 손실
        loss = 0.7 * loss_mse + 0.3 * loss_ssim  # SSIM의 비중을 높임
        
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
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss_mse = criterion_mse(outputs, images)
            loss_ssim = 1 - criterion_ssim(outputs, images)
            loss = 0.7 * loss_mse + 0.3 * loss_ssim
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # 학습률 조정
    scheduler.step(val_loss)

    # Early Stopping 체크
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'fin_vit_unet_autoencoder_model_optimized.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping activated.")
            break
