import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

class WaferDefectClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True):
        super(WaferDefectClassifier, self).__init__()
        
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("지원되지 않는 모델명입니다.")
        
    def forward(self, x):
        return self.model(x)

# 클래스 수 정의 (웨이퍼 결함 유형 수에 따라 조정)
num_classes = 9  # 예: WM-811K 데이터셋의 경우

# 모델 생성
model = WaferDefectClassifier(num_classes, model_name='efficientnet_b0')

# GPU 사용 가능 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 가중치 동결
for param in model.model.parameters():
    param.requires_grad = False

# 마지막 레이어의 가중치만 학습 가능하도록 설정
for param in model.model.classifier.parameters():
    param.requires_grad = True

# 모델 구조 출력
print(model)

# 학습 가능한 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'전체 파라미터 수: {total_params}')
print(f'학습 가능한 파라미터 수: {trainable_params}')

# Focal Loss 구현
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 훈련 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step()

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model = model.state_dict()
                    epochs_no_improve = 0
                    torch.save(best_model, 'best_model.pth')
                else:
                    epochs_no_improve += 1

        if epochs_no_improve == patience:
            print('Early stopping!')
            break

    model.load_state_dict(best_model)
    return model

# 모델 훈련 실행
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=5)