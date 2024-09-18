import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
from itertools import cycle

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_score, classes):
    n_classes = len(classes)
    y_test = np.eye(n_classes)[y_true]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# 모델 평가
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
y_pred, y_true = evaluate_model(model, test_loader, device)

# 클래스 이름 (실제 웨이퍼 결함 유형에 맞게 수정 필요)
class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']

# 혼동 행렬 출력
plot_confusion_matrix(y_true, y_pred, class_names)

# 분류 보고서 출력
print(classification_report(y_true, y_pred, target_names=class_names))

# ROC 곡선 출력
y_score = model(test_loader.dataset.tensors[0].to(device)).detach().cpu().numpy()
plot_roc_curve(y_true, y_score, class_names)

# 교차 검증
cv_scores = cross_val_score(model, test_loader.dataset.tensors[0], y_true, cv=5)
print(f"교차 검증 점수: {cv_scores}")
print(f"평균 교차 검증 점수: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# 오분류된 샘플 분석
misclassified = y_pred != y_true
print(f"오분류된 샘플 수: {misclassified.sum()}")

# 클래스별 성능 분석
for i, class_name in enumerate(class_names):
    class_samples = y_true == i
    class_correct = (y_pred == y_true) & class_samples
    print(f"{class_name}: {class_correct.sum()} / {class_samples.sum()} 정확도: {class_correct.sum() / class_samples.sum():.2f}")