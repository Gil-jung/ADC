import os
from PIL import Image
import imagehash

def check_image_resolution(directory, expected_size=(224, 224)):
    mismatched = []
    for cls in os.listdir(directory):
        class_dir = os.path.join(directory, cls)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path)
                if img.size != expected_size:
                    mismatched.append((img_path, img.size))
            except Exception as e:
                print(f'이미지 열기 실패: {img_path}, 에러: {e}')
    if mismatched:
        print("해상도가 일치하지 않는 이미지들:")
        for path, size in mismatched:
            print(f'{path}: {size}')
    else:
        print("모든 이미지의 해상도가 일치합니다.")

def check_duplicate_images(directory):
    hashes = {}
    duplicates = []
    for cls in os.listdir(directory):
        class_dir = os.path.join(directory, cls)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path)
                img_hash = imagehash.average_hash(img)
                if img_hash in hashes:
                    duplicates.append((img_path, hashes[img_hash]))
                else:
                    hashes[img_hash] = img_path
            except Exception as e:
                print(f'이미지 해시 실패: {img_path}, 에러: {e}')
    if duplicates:
        print("중복 이미지들:")
        for dup, original in duplicates:
            print(f'{dup} 는 {original} 과 중복됩니다.')
    else:
        print("중복된 이미지가 없습니다.")

if __name__ == "__main__":
    train_dir = 'data/split/train'
    print("이미지 해상도 확인:")
    check_image_resolution(train_dir)
    print("\n중복 이미지 확인:")
    check_duplicate_images(train_dir)