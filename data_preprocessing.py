"""
Data preprocessing pipeline for rice disease classification
"""
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

class DatasetPreparator:
    def __init__(self, base_dir, output_dir='processed_dataset'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define categories
        self.healthy_dir = self.base_dir / "Cây lúa khỏe mạnh"
        self.disease_dir = self.base_dir / "II Bệnh gây hại trên lúa"
        self.pest_dir = self.base_dir / "I Côn trùng trên lúa"
        self.nutrition_dir = self.base_dir / "III Thiếu dinh dưỡng"
        
    def create_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15):
        """Create stratified train/val/test splits"""
        print("=" * 80)
        print("CREATING TRAIN/VAL/TEST SPLIT")
        print("=" * 80)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        class_info = {}
        all_files = []
        all_labels = []
        
        # Collect all images with their labels
        # 1. Healthy
        if self.healthy_dir.exists():
            images = list(self.healthy_dir.glob('*.jpg')) + \
                    list(self.healthy_dir.glob('*.JPG')) + \
                    list(self.healthy_dir.glob('*.jpeg'))
            all_files.extend(images)
            all_labels.extend(['Healthy'] * len(images))
            class_info['Healthy'] = len(images)
        
        # 2. Diseases
        if self.disease_dir.exists():
            for disease_subdir in self.disease_dir.iterdir():
                if disease_subdir.is_dir():
                    image_folder = disease_subdir / "Ảnh"
                    if not image_folder.exists():
                        image_folder = disease_subdir / "ảnh"
                    
                    if image_folder.exists():
                        images = list(image_folder.glob('*.jpg')) + \
                                list(image_folder.glob('*.JPG')) + \
                                list(image_folder.glob('*.jpeg'))
                        all_files.extend(images)
                        all_labels.extend([disease_subdir.name] * len(images))
                        class_info[disease_subdir.name] = len(images)
        
        # 3. Pests
        if self.pest_dir.exists():
            for pest_subdir in self.pest_dir.iterdir():
                if pest_subdir.is_dir():
                    image_folder = pest_subdir / "Ảnh"
                    if not image_folder.exists():
                        image_folder = pest_subdir / "ảnh"
                    
                    if image_folder.exists():
                        images = list(image_folder.glob('*.jpg')) + \
                                list(image_folder.glob('*.JPG')) + \
                                list(image_folder.glob('*.jpeg'))
                        all_files.extend(images)
                        all_labels.extend([pest_subdir.name] * len(images))
                        class_info[pest_subdir.name] = len(images)
        
        # 4. Nutrition
        if self.nutrition_dir.exists():
            for nutrition_subdir in self.nutrition_dir.iterdir():
                if nutrition_subdir.is_dir():
                    for sub_folder in nutrition_subdir.iterdir():
                        if sub_folder.is_dir():
                            images = list(sub_folder.glob('*.jpg')) + \
                                    list(sub_folder.glob('*.JPG')) + \
                                    list(sub_folder.glob('*.jpeg'))
                            
                            if len(images) > 0:
                                all_files.extend(images)
                                all_labels.extend([nutrition_subdir.name] * len(images))
                                class_info[nutrition_subdir.name] = len(images)
        
        print(f"\nTotal images: {len(all_files)}")
        print(f"Total classes: {len(class_info)}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'file_path': all_files,
            'label': all_labels
        })
        
        # Stratified split
        train_df, temp_df = train_test_split(
            df, test_size=(val_size + test_size), 
            stratify=df['label'], random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(test_size / (val_size + test_size)),
            stratify=temp_df['label'], random_state=42
        )
        
        print(f"\nTrain samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Copy files to respective directories
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f"\nCopying {split_name} files...")
            for label in split_df['label'].unique():
                class_dir = self.output_dir / split_name / label
                class_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, row in split_df.iterrows():
                src = row['file_path']
                dst_dir = self.output_dir / split_name / row['label']
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                
                if (idx + 1) % 1000 == 0:
                    print(f"  Copied {idx + 1}/{len(split_df)} files...")
        
        # Save metadata
        metadata = {
            'total_images': len(all_files),
            'total_classes': len(class_info),
            'class_distribution': class_info,
            'split_sizes': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            }
        }
        
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save split CSVs
        train_df.to_csv(self.output_dir / 'train_split.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_split.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_split.csv', index=False)
        
        print("\n✓ Dataset split complete!")
        return train_df, val_df, test_df


if __name__ == "__main__":
    base_dir = r"C:\Users\admin\Downloads\DATASET"
    
    preparator = DatasetPreparator(base_dir)
    train_df, val_df, test_df = preparator.create_train_val_test_split()