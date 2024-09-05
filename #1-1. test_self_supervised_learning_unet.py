import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from transformers import ViTModel

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Normalize의 반대 연산을 위한 함수 정의
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Autoencoder 모델 정의 (저장된 모델과 동일한 구조)
class ViTUAutoencoder(nn.Module):
    def __init__(self):
        super(ViTUAutoencoder, self).__init__()
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

# 모델 로드
autoencoder = ViTUAutoencoder().to(device)
autoencoder.load_state_dict(torch.load('fin_vit_unet_autoencoder_model_optimized.pth'))  # 학습된 모델 가중치 로드
autoencoder.eval()

# 테스트 이미지 로드 및 전처리
image_path = 'C:/Users/IIPL02/Desktop/LIQE_nonIndex/# 1. 지도학습/DatabaseImage0582.JPG'
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0).to(device)

# 복원된 이미지 생성
with torch.no_grad():
    reconstructed_image = autoencoder(input_image)

# 시각화 함수
def imshow(tensor, title=None, denorm=False):
    image = tensor.cpu().clone().squeeze(0)
    
    # Denormalization 적용
    if denorm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = denormalize(image, mean, std)
    
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

# 원본 이미지와 복원된 이미지 비교
imshow(input_image, title="Original Image", denorm=True)  # 원본 이미지 출력 (Denormalize 적용)
imshow(reconstructed_image, title="Reconstructed Image")  # 복원된 이미지 출력

"""
Denormalization 함수 추가: denormalize 함수를 사용해 정규화된 이미지를 원래 픽셀 값으로 복원합니다.
imshow 함수 수정: denorm 플래그를 추가하여 정규화된 이미지를 복원할 수 있도록 했습니다.
이미지 출력 시 Denormalize 적용: 원본 이미지를 시각화할 때 denormalize를 적용하여 왜곡 없이 이미지를 볼 수 있도록 했습니다.
"""
