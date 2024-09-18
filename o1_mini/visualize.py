import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_sample_images(directory, classes, samples_per_class=5):
    plt.figure(figsize=(samples_per_class * 3, len(classes) * 3))
    
    for i, cls in enumerate(classes):
        class_dir = os.path.join(directory, cls)
        images = os.listdir(class_dir)
        sampled_images = images[:samples_per_class]
        
        for j, img_name in enumerate(sampled_images):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                plt_idx = i * samples_per_class + j + 1
                plt.subplot(len(classes), samples_per_class, plt_idx)
                plt.imshow(img)
                plt.axis('off')
                if j == 0:
                    plt.ylabel(cls, fontsize=12)
            except Exception as e:
                print(f'이미지 로드 실패: {img_path}, 에러: {e}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'sample_images.png'))
    plt.show()

if __name__ == "__main__":
    train_dir = 'data/split/train'
    classes = os.listdir(train_dir)
    visualize_sample_images(train_dir, classes, samples_per_class=5)