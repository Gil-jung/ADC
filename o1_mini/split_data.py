import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir, train_dir, val_dir, test_dir, test_size=0.1, val_size=0.1):
    classes = os.listdir(source_dir)
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        images = os.listdir(class_dir)
        train_val_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        train_images, val_images = train_test_split(train_val_images, test_size=val_size, random_state=42)
        
        # 학습 데이터 복사
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, cls, img))
        
        # 검증 데이터 복사
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, cls, img))
        
        # 테스트 데이터 복사
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, cls, img))
    
    print('데이터 분할 완료')

if __name__ == "__main__":
    split_data(
        source_dir='data/encoded_images/train',
        train_dir='data/split/train',
        val_dir='data/split/validation',
        test_dir='data/split/test'
    )