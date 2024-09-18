import os
from PIL import Image

def remove_corrupted_images(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(file_path)
            img.verify()  # 이미지 파일 검증
            img = Image.open(file_path)  # 다시 열기
            img = img.convert('RGB')  # RGB 변환
            img.save(os.path.join(output_dir, filename))
            print(f'정상 이미지 저장: {filename}')
        except (IOError, SyntaxError) as e:
            print(f'손상된 이미지 삭제: {filename}')

if __name__ == "__main__":
    remove_corrupted_images('data/raw_images', 'data/clean_images')