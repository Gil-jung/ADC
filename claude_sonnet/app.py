import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import logging
import uvicorn
from typing import List

# 모델 로드 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('best_model.pth', map_location=device)
model.eval()

# 전처리 함수
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# 클래스 이름 (실제 웨이퍼 결함 유형에 맞게 수정 필요)
class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']

# FastAPI 앱 설정
app = FastAPI(title="Wafer Defect Classifier API")

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    tensor = preprocess_image(img_bytes)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
    
    result = {
        'class_name': class_name,
        'confidence': confidence
    }
    
    # 로깅
    logging.info(f"Prediction: {result}")
    
    return JSONResponse(content=result)

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    tensors = []
    for file in files:
        img_bytes = await file.read()
        tensor = preprocess_image(img_bytes)
        tensors.append(tensor)
    
    batch_tensor = torch.cat(tensors, dim=0)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted = torch.max(outputs, 1)
        confidences = torch.nn.functional.softmax(outputs, dim=1)
    
    results = []
    for i in range(len(files)):
        result = {
            'class_name': class_names[predicted[i].item()],
            'confidence': confidences[i][predicted[i]].item()
        }
        results.append(result)
    
    # 로깅
    logging.info(f"Batch Prediction: {results}")
    
    return JSONResponse(content=results)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)