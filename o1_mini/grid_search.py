import torch
from torch import nn
from torch.optim import Adam
from train import train_model
from model import WaferDefectClassifier
from data_loader import get_data_loaders
from train_utils import get_loss_criterion, get_scheduler
from early_stopping import EarlyStopping

def perform_grid_search(param_grid, train_dir, val_dir, test_dir, device, num_classes, num_epochs=25, save_path='models/best_model.pth'):
    best_acc = 0.0
    best_params = {}
    results = []

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            print(f'\n하이퍼파라미터 설정: 학습률={lr}, 배치 크기={batch_size}')
            
            # 데이터 로더 준비
            train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir, batch_size, image_size=(224, 224))
            
            # 모델 초기화
            model = WaferDefectClassifier(num_classes=num_classes, pretrained=True).to(device)
            
            # 손실 함수, 옵티마이저, 스케줄러 설정
            criterion = get_loss_criterion()
            optimizer = Adam(model.parameters(), lr=lr)
            scheduler = get_scheduler(optimizer, step_size=7, gamma=0.1)
            
            # 조기 종료 설정
            early_stopping = EarlyStopping(patience=5, verbose=True)
            
            # 모델 학습
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=num_epochs,
                save_path=save_path,
                patience=5
            )
            
            # 검증 정확도 평가
            val_acc = evaluate_model_accuracy(model, val_loader, device)
            results.append({'learning_rate': lr, 'batch_size': batch_size, 'val_accuracy': val_acc})
            
            # 베스트 모델 업데이트
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                torch.save(model.state_dict(), save_path)
                print(f'최고 성능 업데이트: 정확도={val_acc:.4f}')
    
    print(f'\n그리드 서치 완료. 최고 정확도={best_acc:.4f}, 하이퍼파라미터={best_params}')
    return results, best_params

def evaluate_model_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

if __name__ == "__main__":
    # 하이퍼파라미터 그리드 설정
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64]
    }
    
    # 기타 설정
    num_classes = 5  # 결함 유형 수
    num_epochs = 25
    train_dir = 'data/split/train'
    val_dir = 'data/split/validation'
    test_dir = 'data/split/test'
    save_path = 'models/best_model_grid_search.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 그리드 서치 수행
    results, best_params = perform_grid_search(
        param_grid=param_grid,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        device=device,
        num_classes=num_classes,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    # 결과 출력
    for res in results:
        print(res)