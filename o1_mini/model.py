import torch
import torch.nn as nn
from torchvision import models

class WaferDefectClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(WaferDefectClassifier, self).__init__()
        # 사전 학습된 ResNet-50 모델 로드
        self.model = models.resnet50(pretrained=pretrained)
        # 기존의 fc (fully connected) 층을 새로운 층으로 교체
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class WaferDefectClassifierWithDropout(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_p=0.5):
        super(WaferDefectClassifierWithDropout, self).__init__()
        # 사전 학습된 ResNet-50 모델 로드
        self.model = models.resnet50(pretrained=pretrained)
        # 기존의 fc (fully connected) 층을 새로운 층으로 교체
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def initialize_model(num_classes, device):
    model = WaferDefectClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    return model

def print_model_summary(model, input_size=(3, 224, 224)):
    from torchsummary import summary
    summary(model, input_size)

if __name__ == "__main__":
    num_classes = 5  # 결함 유형의 수에 맞게 조정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(num_classes, device)
    print_model_summary(model)