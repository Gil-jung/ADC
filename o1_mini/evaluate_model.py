import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import WaferDefectClassifier
from data_loader import get_data_loaders

def load_model(save_path, num_classes, device):
    model = WaferDefectClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device, class_names, report_path='reports/classification_report.txt', confusion_matrix_path='reports/confusion_matrix.png'):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 분류 보고서 생성
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'분류 보고서 저장: {report_path}')

    # 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('예측 레이블')
    plt.ylabel('실제 레이블')
    plt.title('혼동 행렬')
    plt.savefig(confusion_matrix_path)
    plt.show()
    print(f'혼동 행렬 저장: {confusion_matrix_path}')

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    num_classes = 5  # 결함 유형 수에 맞게 설정
    batch_size = 32
    image_size = (224, 224)
    model_path = 'models/best_model.pth'
    report_path = 'reports/classification_report.txt'
    confusion_matrix_path = 'reports/confusion_matrix.png'

    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'사용 중인 디바이스: {device}')

    # 데이터 로더 준비
    train_dir = 'data/split/train'
    val_dir = 'data/split/validation'
    test_dir = 'data/split/test'
    _, _, test_loader = get_data_loaders(train_dir, val_dir, test_dir, batch_size, image_size)

    # 모델 로드
    model = load_model(model_path, num_classes, device)

    # 클래스 이름 설정
    class_names = test_loader.dataset.classes

    # 모델 평가
    evaluate_model(model, test_loader, device, class_names, report_path, confusion_matrix_path)