import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# 이미지 전처리를 위한 변환 정의
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터 증강을 위한 변환 정의
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

def load_and_preprocess_image(image_path, augment=True):
    image = Image.open(image_path).convert('RGB')
    if augment:
        image = data_augmentation(image)  # 데이터 증강 적용
    image = preprocess(image)
    return image

# 이미지 로드 및 전처리
def load_dataset(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = load_and_preprocess_image(image_path)
            images.append(image)
            labels.append(int(label))
    return images, labels

# 데이터셋 로드
data_dir = 'path/to/WM-811K_dataset'
images, labels = load_dataset(data_dir)

# 데이터셋 분할 (훈련, 검증, 테스트)
X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)

# 텐서로 변환
X_train = torch.stack(X_train)
X_val = torch.stack(X_val)
X_test = torch.stack(X_test)
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)

print(f"훈련 데이터 크기: {X_train.shape}")
print(f"검증 데이터 크기: {X_val.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")

class WaferDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 데이터 증강을 위한 변환 정의
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 생성
train_dataset = WaferDataset(X_train, y_train, transform=train_transform)
val_dataset = WaferDataset(X_val, y_val, transform=val_transform)
test_dataset = WaferDataset(X_test, y_test, transform=val_transform)

# 클래스 불균형 처리를 위한 가중치 계산
class_counts = Counter(y_train.numpy())
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in y_train.numpy()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 데이터 로더 확인
for images, labels in train_loader:
    print(f"배치 크기: {images.shape}")
    print(f"레이블: {labels}")
    break

print(f"훈련 데이터 배치 수: {len(train_loader)}")
print(f"검증 데이터 배치 수: {len(val_loader)}")
print(f"테스트 데이터 배치 수: {len(test_loader)}")