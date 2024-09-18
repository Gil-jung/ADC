import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(directory):
    classes = os.listdir(directory)
    class_counts = {cls: len(os.listdir(os.path.join(directory, cls))) for cls in classes}
    
    # 막대 그래프 생성
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xlabel('클래스')
    plt.ylabel('이미지 수')
    plt.title('클래스별 이미지 분포')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'class_distribution.png'))
    plt.show()
    
    # 클래스 불균형 지표 출력
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    imbalance_ratio = class_counts[max_class] / class_counts[min_class]
    
    print(f'가장 많은 클래스: {max_class} ({class_counts[max_class]}개)')
    print(f'가장 적은 클래스: {min_class} ({class_counts[min_class]}개)')
    print(f'클래스 불균형 비율: {imbalance_ratio:.2f}')

if __name__ == "__main__":
    train_dir = 'data/split/train'
    plot_class_distribution(train_dir)