import torch
from model import WaferDefectClassifier

def load_deployed_model(save_path, num_classes, device):
    model = WaferDefectClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model