import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from load_model_deployment import load_deployed_model
from torchvision import transforms

# FastAPI 앱 초기화
app = FastAPI(title="Wafer Defect Classification API")

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 로드
num_classes = 5  # 결함 유형 수
model_path = 'models/best_model.pth'
model = load_deployed_model(model_path, num_classes, device)

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 클래스 이름 정의 (예시)
class_names = ["Class A", "Class B", "Class C", "Class D", "Class E"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 이미지 파일 읽기
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 예측 수행
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
    
    return {"predicted_class": predicted_class}