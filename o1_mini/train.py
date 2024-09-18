import os
import torch
from torch import nn
from tqdm import tqdm
from model import WaferDefectClassifier
from data_loader import get_data_loaders
from train_utils import get_loss_criterion, get_optimizer, get_scheduler
from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, save_path='models/best_model.pth', patience=5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_accuracy = 0.0

    # TensorBoard Writer 설정
    writer = SummaryWriter('runs/wafer_defect_classification')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 에포크마다 학습 단계와 검증 단계 수행
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
                dataloader = train_loader
            else:
                model.eval()   # 모델을 평가 모드로 설정
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 데이터 순회
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계에서만 역전파 및 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # TensorBoard에 로그 기록
            if phase == 'train':
                writer.add_scalar('Loss/Train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/Val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/Val', epoch_acc, epoch)

            # 검증 단계일 때 체크포인트 및 조기 종료 체크
            if phase == 'val':
                early_stopping(epoch_acc, model, save_path)
                if early_stopping.early_stop:
                    print("조기 종료 트리거됨")
                    writer.close()
                    return

    print(f'학습 완료. 최고 검증 정확도: {best_accuracy:.4f}')
    writer.close()

def calculate_class_weights(train_dir):
    classes = os.listdir(train_dir)
    class_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in classes]
    total = sum(class_counts)
    class_weights = [total / count for count in class_counts]
    class_weights = torch.FloatTensor(class_weights)
    return class_weights

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    num_classes = 5  # 결함 유형 수에 맞게 설정
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    image_size = (224, 224)
    save_path = 'models/best_model_class_weights.pth'
    patience = 5  # 조기 종료를 위한 patience 설정

    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'사용 중인 디바이스: {device}')

    # 데이터 로더 준비
    train_dir = 'data/split/train'
    val_dir = 'data/split/validation'
    test_dir = 'data/split/test'
    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir, batch_size, image_size)

    # 모델 초기화
    model = WaferDefectClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # 클래스 가중치 계산
    class_weights = calculate_class_weights(train_dir).to(device)
    print(f'클래스 가중치: {class_weights}')

    # 손실 함수, 옵티마이저, 스케줄러 설정
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer)

    # 모델 학습
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path, patience)

    # 모델 저장
    save_model(model, 'models/final_model.pth')