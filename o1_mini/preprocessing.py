import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def preprocess_and_augment_images(source_dir, target_dir, image_size=(224, 224)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 일반적인 이미지 정규화 값
                             std=[0.229, 0.224, 0.225])
    ])
    
    classes = os.listdir(source_dir)
    
    for cls in classes:
        class_source_dir = os.path.join(source_dir, cls)
        class_target_dir = os.path.join(target_dir, cls)
        if not os.path.exists(class_target_dir):
            os.makedirs(class_target_dir)
        
        images = os.listdir(class_source_dir)
        for img_name in tqdm(images, desc=f'Processing {cls}'):
            img_path = os.path.join(class_source_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                transformed_image = transform(image)
                
                # 텐서에서 이미지를 다시 PIL 이미지로 변환
                inv_normalize = transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                unnormalized_image = inv_normalize(transformed_image)
                unnormalized_image = transforms.ToPILImage()(unnormalized_image)
                
                # 증강된 이미지 저장
                augmented_img_name = f'aug_{img_name}'
                augmented_img_path = os.path.join(class_target_dir, augmented_img_name)
                unnormalized_image.save(augmented_img_path)
                
            except Exception as e:
                print(f'이미지 처리 실패: {img_path}, 에러: {e}')
    
    print('이미지 전처리 및 증강 완료')

if __name__ == "__main__":
    preprocess_and_augment_images('data/clean_images', 'data/preprocessed_images')