"""
PyTorch data loaders with augmentation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd

class RiceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Create label mapping
        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.class_to_idx[label]
        return image, label_idx


def get_transforms(img_size=224):
    """Get train and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir='processed_dataset', batch_size=32, num_workers=4):
    """Create train, val, test data loaders"""
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = RiceDataset(
        f'{data_dir}/train_split.csv',
        transform=train_transform
    )
    
    val_dataset = RiceDataset(
        f'{data_dir}/val_split.csv',
        transform=val_transform
    )
    
    test_dataset = RiceDataset(
        f'{data_dir}/test_split.csv',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    # Test data loaders
    train_loader, val_loader, test_loader, classes = create_data_loaders(batch_size=16)
    
    print(f"Number of classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")