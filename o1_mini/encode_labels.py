import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import shutil

def encode_labels(source_dir, encoded_dir):
    if not os.path.exists(encoded_dir):
        os.makedirs(encoded_dir)
    
    label_encoder = LabelEncoder()
    classes = os.listdir(source_dir)
    label_encoder.fit(classes)
    
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        target_class_dir = os.path.join(encoded_dir, str(label_encoder.transform([cls])[0]))
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)
        for filename in os.listdir(class_dir):
            shutil.copy(os.path.join(class_dir, filename), os.path.join(target_class_dir, filename))
    
    # 레이블 매핑 저장
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    with open(os.path.join(encoded_dir, 'label_mapping.csv'), 'w') as f:
        for key, value in label_mapping.items():
            f.write(f'{key},{value}\n')
    print('레이블 인코딩 완료 및 매핑 저장')

if __name__ == "__main__":
    encode_labels('data/preprocessed_images/train', 'data/encoded_images/train')
    encode_labels('data/preprocessed_images/validation', 'data/encoded_images/validation')