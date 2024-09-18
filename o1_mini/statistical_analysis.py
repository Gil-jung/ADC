import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def statistical_analysis(directory):
    classes = os.listdir(directory)
    data = []
    
    for cls in classes:
        class_dir = os.path.join(directory, cls)
        image_names = os.listdir(class_dir)
        image_count = len(image_names)
        data.append({'클래스': cls, '이미지 수': image_count})
    
    df = pd.DataFrame(data)
    
    # 기본 통계량 출력
    print("기본 통계량:")
    print(df.describe())
    
    # 클래스별 이미지 수 출력
    print("\n클래스별 이미지 수:")
    print(df)
    
    # 박스 플롯 생성
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='클래스', y='이미지 수', data=df)
    plt.xticks(rotation=45)
    plt.title('클래스별 이미지 수 분포')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'class_boxplot.png'))
    plt.show()

if __name__ == "__main__":
    train_dir = 'data/split/train'
    statistical_analysis(train_dir)