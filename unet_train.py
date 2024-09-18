import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms

# Attention 메커니즘
class AttentionModule(nn.Module):
    def __init__(self, filters):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 1)
        self.conv2 = nn.Conv2d(filters, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

# U-Net 기반 세그멘테이션 모델
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Decoder
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        
        self.attention = AttentionModule(64)
        self.final = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.up(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
        
        attention = self.attention(d2)
        out = self.final(attention)
        
        return torch.sigmoid(out)

# 이미지 augmentation 함수
def augment_image(image, mask):
    # 회전
    angle = np.random.uniform(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))
    
    # 좌우 뒤집기
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    # 밝기 조정
    image = image.astype(np.float32)
    image = image * np.random.uniform(0.8, 1.2)
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image, mask

# 데이터셋 클래스
class WaferDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, self.img_size)
        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, self.img_size)
        
        img, mask = augment_image(img, mask)
        
        img = self.transform(img)
        mask = torch.from_numpy(mask).float() / 255.0
        mask = mask.unsqueeze(0)
        
        return img, mask

# 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터 경로 설정 (실제 경로로 변경 필요)
    train_image_paths = ['train_image_1.jpg', 'train_image_2.jpg', ...]
    train_mask_paths = ['train_mask_1.png', 'train_mask_2.png', ...]
    val_image_paths = ['val_image_1.jpg', 'val_image_2.jpg', ...]
    val_mask_paths = ['val_mask_1.png', 'val_mask_2.png', ...]
    
    # 하이퍼파라미터 설정
    img_size = (256, 256)
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    
    # 데이터셋 및 데이터로더 초기화
    train_dataset = WaferDataset(train_image_paths, train_mask_paths, img_size)
    val_dataset = WaferDataset(val_image_paths, val_mask_paths, img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 모델, 손실 함수, 옵티마이저 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # 모델 저장
    torch.save(model.state_dict(), 'best_model.pth')