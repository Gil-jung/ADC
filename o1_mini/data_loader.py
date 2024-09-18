import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, val_dir, test_dir, batch_size=32, image_size=(224, 224)):
    # 데이터 전처리 및 증강 설정
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def get_advanced_data_loaders(train_dir, val_dir, test_dir, batch_size=32, image_size=(224, 224)):
    # 데이터 전처리 및 증강 설정
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def get_data_loaders_with_sampler(train_dir, val_dir, test_dir, batch_size=32, image_size=(224, 224)):
    # 데이터 전처리 설정
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    # 클래스별 샘플 수 계산
    class_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in train_dataset.classes]
    class_weights = [1.0 / count for count in class_counts]
    samples_weights = [class_weights[label] for _, label in train_dataset]

    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_dir = 'data/split/train'
    val_dir = 'data/split/validation'
    test_dir = 'data/split/test'
    # train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir, batch_size=32, image_size=(224, 224))
    train_loader, val_loader, test_loader = get_advanced_data_loaders(train_dir, val_dir, test_dir, batch_size=32, image_size=(224, 224))
    # train_loader, val_loader, test_loader = get_data_loaders_with_sampler(train_dir, val_dir, test_dir, batch_size=32, image_size=(224, 224))