import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
from torchvision import transforms

def get_gradcam(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
    return grayscale_cam[0, :]

def visualize_gradcam(model, img_tensor, class_names, target_layer):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 원본 이미지로 변환
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    # Grad-CAM 생성
    grayscale_cam = get_gradcam(model, img_tensor, target_layer)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    # 예측 클래스 확인
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    
    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'Grad-CAM: {predicted_class}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def shap_analysis(model, background_data, test_sample, class_names):
    e = shap.DeepExplainer(model, background_data)
    shap_values = e.shap_values(test_sample)
    
    shap.image_plot(shap_values, test_sample.cpu().numpy(), show=False)
    plt.suptitle("SHAP values for each class")
    plt.tight_layout()
    plt.show()

# Grad-CAM 시각화
target_layer = model.model.features[-1]  # EfficientNet의 경우, 마지막 특성 레이어 선택
sample_img, _ = next(iter(test_loader))
visualize_gradcam(model, sample_img[0], class_names, target_layer)

# SHAP 분석
background = next(iter(train_loader))[0][:100].to(device)  # 배경 데이터로 훈련 세트의 일부 사용
test_samples = next(iter(test_loader))[0][:10].to(device)  # 테스트 세트의 일부 샘플 선택
shap_analysis(model, background, test_samples, class_names)

# 특성 중요도 분석 (EfficientNet의 경우 복잡하므로 생략)

# 오분류된 샘플 시각화
def visualize_misclassified(model, test_loader, class_names, num_samples=5):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                if preds[i] != labels[i]:
                    misclassified.append((inputs[i], labels[i], preds[i]))
                    if len(misclassified) == num_samples:
                        break
            if len(misclassified) == num_samples:
                break
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i, (img, true_label, pred_label) in enumerate(misclassified):
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[true_label.item()]}\nPred: {class_names[pred_label.item()]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_misclassified(model, test_loader, class_names)