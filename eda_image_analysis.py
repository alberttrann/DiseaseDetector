import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(0)

class RiceImageAnalyzer:
    def __init__(self, base_dir):
        """Initialize with base directory containing image folders"""
        self.base_dir = Path(base_dir)
        self.output_dir = Path('eda_outputs/image_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define categories based on actual structure
        self.healthy_dir = self.base_dir / "Cây lúa khỏe mạnh"
        self.disease_dir = self.base_dir / "II Bệnh gây hại trên lúa"
        self.pest_dir = self.base_dir / "I Côn trùng trên lúa"
        self.nutrition_dir = self.base_dir / "III Thiếu dinh dưỡng"
        
        # Track errors
        self.read_errors = []
        
    def read_image_safe(self, img_path):
        """
        Safely read image with fallback methods for Unicode paths
        Returns: numpy array (RGB) or None
        """
        try:
            # Method 1: Use PIL (best for Unicode paths)
            with Image.open(str(img_path)) as img:
                img_array = np.array(img)
                
                # Convert to RGB if needed
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] == 3:  # Already RGB/BGR
                    # PIL loads as RGB, no conversion needed
                    pass
                
                return img_array
            
        except Exception as e:
            self.read_errors.append({
                'path': str(img_path),
                'error': str(e)
            })
            return None
    
    def scan_directories(self):
        """Scan all directories and collect image information"""
        print("=" * 80)
        print("SCANNING IMAGE DIRECTORIES")
        print("=" * 80)
        
        self.image_data = defaultdict(list)
        self.class_info = {}
        
        total_images = 0
        
        # 1. Scan Healthy Rice (direct images in folder)
        if self.healthy_dir.exists():
            print(f"\nScanning: Healthy Rice")
            print("-" * 80)
            
            image_files = list(self.healthy_dir.glob('*.jpg')) + \
                         list(self.healthy_dir.glob('*.jpeg')) + \
                         list(self.healthy_dir.glob('*.JPG')) + \
                         list(self.healthy_dir.glob('*.png'))
            
            num_images = len(image_files)
            total_images += num_images
            
            self.class_info['Healthy'] = {
                'category': 'Healthy',
                'count': num_images,
                'path': self.healthy_dir,
                'images': image_files
            }
            
            print(f"  Healthy: {num_images} images")
        
        # 2. Scan Diseases (subdirectories with Ảnh folder)
        if self.disease_dir.exists():
            print(f"\nScanning: Diseases")
            print("-" * 80)
            
            for disease_subdir in self.disease_dir.iterdir():
                if disease_subdir.is_dir():
                    # Look for Ảnh or ảnh subfolder
                    image_folder = disease_subdir / "Ảnh"
                    if not image_folder.exists():
                        image_folder = disease_subdir / "ảnh"
                    
                    if image_folder.exists():
                        image_files = list(image_folder.glob('*.jpg')) + \
                                    list(image_folder.glob('*.jpeg')) + \
                                    list(image_folder.glob('*.JPG')) + \
                                    list(image_folder.glob('*.png'))
                        
                        num_images = len(image_files)
                        total_images += num_images
                        
                        class_name = disease_subdir.name
                        self.class_info[class_name] = {
                            'category': 'Disease',
                            'count': num_images,
                            'path': image_folder,
                            'images': image_files
                        }
                        
                        print(f"  {class_name}: {num_images} images")
        
        # 3. Scan Pests (subdirectories with Ảnh folder)
        if self.pest_dir.exists():
            print(f"\nScanning: Pests")
            print("-" * 80)
            
            for pest_subdir in self.pest_dir.iterdir():
                if pest_subdir.is_dir():
                    # Look for Ảnh or ảnh subfolder
                    image_folder = pest_subdir / "Ảnh"
                    if not image_folder.exists():
                        image_folder = pest_subdir / "ảnh"
                    
                    if image_folder.exists():
                        image_files = list(image_folder.glob('*.jpg')) + \
                                    list(image_folder.glob('*.jpeg')) + \
                                    list(image_folder.glob('*.JPG')) + \
                                    list(image_folder.glob('*.png'))
                        
                        num_images = len(image_files)
                        total_images += num_images
                        
                        class_name = pest_subdir.name
                        self.class_info[class_name] = {
                            'category': 'Pest',
                            'count': num_images,
                            'path': image_folder,
                            'images': image_files
                        }
                        
                        print(f"  {class_name}: {num_images} images")
        
        # 4. Scan Nutrition Deficiency (subdirectories with subfolder)
        if self.nutrition_dir.exists():
            print(f"\nScanning: Nutrition Deficiency")
            print("-" * 80)
            
            for nutrition_subdir in self.nutrition_dir.iterdir():
                if nutrition_subdir.is_dir():
                    # Look for subfolders with images
                    for sub_folder in nutrition_subdir.iterdir():
                        if sub_folder.is_dir():
                            image_files = list(sub_folder.glob('*.jpg')) + \
                                        list(sub_folder.glob('*.jpeg')) + \
                                        list(sub_folder.glob('*.JPG')) + \
                                        list(sub_folder.glob('*.png'))
                            
                            if len(image_files) > 0:
                                num_images = len(image_files)
                                total_images += num_images
                                
                                class_name = f"{nutrition_subdir.name}"
                                self.class_info[class_name] = {
                                    'category': 'Nutrition',
                                    'count': num_images,
                                    'path': sub_folder,
                                    'images': image_files
                                }
                                
                                print(f"  {class_name}: {num_images} images")
        
        print(f"\n{'='*80}")
        print(f"Total Images Found: {total_images}")
        print(f"Total Classes: {len(self.class_info)}")
        print(f"{'='*80}\n")
        
        return total_images
        
    def analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        print("\n" + "=" * 80)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # Create DataFrame
        data = []
        for class_name, info in self.class_info.items():
            data.append({
                'Class': class_name,
                'Category': info['category'],
                'Count': info['count']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Count', ascending=False)
        
        print("\nClass Distribution:")
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv(self.output_dir / 'class_distribution.csv', index=False)
        
        # Calculate category statistics
        category_stats = df.groupby('Category').agg({
            'Count': ['sum', 'mean', 'min', 'max', 'count']
        }).round(2)
        print("\n" + "-" * 80)
        print("Category Statistics:")
        print("-" * 80)
        print(category_stats)
        
        # Visualizations
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall distribution by category (Pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        category_counts = df.groupby('Category')['Count'].sum()
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        ax1.pie(category_counts.values, labels=category_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Distribution by Main Category', fontsize=14, fontweight='bold')
        
        # 2. Bar chart of all classes
        ax2 = fig.add_subplot(gs[0, 1:])
        top_classes = df.head(20)
        bars = ax2.barh(range(len(top_classes)), top_classes['Count'])
        
        # Color bars by category
        category_colors = {
            'Healthy': '#2ecc71',
            'Disease': '#e74c3c', 
            'Pest': '#f39c12',
            'Nutrition': '#3498db'
        }
        for idx, (_, row) in enumerate(top_classes.iterrows()):
            bars[idx].set_color(category_colors.get(row['Category'], '#95a5a6'))
        
        ax2.set_yticks(range(len(top_classes)))
        ax2.set_yticklabels([name[:50] for name in top_classes['Class']], fontsize=8)
        ax2.set_xlabel('Number of Images', fontsize=11)
        ax2.set_title('Top 20 Classes by Image Count', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat) 
                          for cat, color in category_colors.items()]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        # 3. Category-wise stacked bar
        ax3 = fig.add_subplot(gs[1, 0])
        for idx, category in enumerate(df['Category'].unique()):
            cat_data = df[df['Category'] == category]
            ax3.bar(category, cat_data['Count'].sum(), 
                   color=category_colors.get(category, '#95a5a6'))
        
        ax3.set_ylabel('Total Images', fontsize=11)
        ax3.set_title('Total Images by Category', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='x', rotation=0)
        
        # Add value labels on bars
        for idx, category in enumerate(df['Category'].unique()):
            cat_sum = df[df['Category'] == category]['Count'].sum()
            ax3.text(idx, cat_sum + 100, f'{cat_sum:,}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. Box plot of image counts by category
        ax4 = fig.add_subplot(gs[1, 1])
        categories = df['Category'].unique()
        data_for_box = [df[df['Category'] == cat]['Count'].values for cat in categories]
        bp = ax4.boxplot(data_for_box, labels=categories, patch_artist=True)
        
        for patch, category in zip(bp['boxes'], categories):
            patch.set_facecolor(category_colors.get(category, '#95a5a6'))
        
        ax4.set_ylabel('Images per Class', fontsize=11)
        ax4.set_title('Distribution of Images per Class (by Category)', 
                     fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        ax4.tick_params(axis='x', rotation=0)
        
        # 5. Number of classes per category
        ax5 = fig.add_subplot(gs[1, 2])
        class_counts = df['Category'].value_counts()
        bars = ax5.bar(class_counts.index, class_counts.values)
        for idx, category in enumerate(class_counts.index):
            bars[idx].set_color(category_colors.get(category, '#95a5a6'))
        
        ax5.set_ylabel('Number of Classes', fontsize=11)
        ax5.set_title('Number of Classes per Category', fontsize=14, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for idx, (category, count) in enumerate(class_counts.items()):
            ax5.text(idx, count + 0.1, str(count), 
                    ha='center', va='bottom', fontweight='bold')
        
        # 6. Disease breakdown
        ax6 = fig.add_subplot(gs[2, :])
        disease_df = df[df['Category'] == 'Disease'].sort_values('Count', ascending=True)
        if len(disease_df) > 0:
            bars = ax6.barh(range(len(disease_df)), disease_df['Count'])
            for bar in bars:
                bar.set_color('#e74c3c')
            ax6.set_yticks(range(len(disease_df)))
            ax6.set_yticklabels([name[:60] for name in disease_df['Class']], fontsize=9)
            ax6.set_xlabel('Number of Images', fontsize=11)
            ax6.set_title('All Disease Classes', fontsize=14, fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
        
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n Saved: class_distribution.png")
        plt.close()
        
    def analyze_image_properties(self, sample_size=50):
        """Analyze image properties (dimensions, colors, etc.)"""
        print("\n" + "=" * 80)
        print(f"IMAGE PROPERTIES ANALYSIS (Sampling {sample_size} images per class)")
        print("=" * 80)
        
        properties = {
            'heights': [],
            'widths': [],
            'aspects': [],
            'channels': [],
            'file_sizes': [],
            'mean_brightness': [],
            'std_brightness': [],
            'class': [],
            'category': []
        }
        
        # Sample images from each class
        sampled_count = 0
        failed_count = 0
        
        print("Processing images...", end='', flush=True)
        
        for class_name, info in self.class_info.items():
            images = info['images']
            sample = np.random.choice(images, min(sample_size, len(images)), replace=False)
            
            for img_path in sample:
                try:
                    # Use safe image reading (PIL only - no OpenCV warnings)
                    img = self.read_image_safe(img_path)
                    
                    if img is None:
                        failed_count += 1
                        continue
                    
                    h, w = img.shape[:2]
                    c = img.shape[2] if len(img.shape) == 3 else 1
                    
                    properties['heights'].append(h)
                    properties['widths'].append(w)
                    properties['aspects'].append(w / h)
                    properties['channels'].append(c)
                    properties['file_sizes'].append(os.path.getsize(img_path) / 1024)  # KB
                    properties['class'].append(class_name)
                    properties['category'].append(info['category'])
                    
                    # Brightness statistics (using numpy instead of cv2)
                    if len(img.shape) == 3:
                        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])  # RGB to grayscale formula
                    else:
                        gray = img
                    
                    properties['mean_brightness'].append(np.mean(gray))
                    properties['std_brightness'].append(np.std(gray))
                    
                    sampled_count += 1
                    
                    # Progress indicator
                    if sampled_count % 100 == 0:
                        print(".", end='', flush=True)
                    
                except Exception as e:
                    failed_count += 1
                    continue
        
        print(" Done!")
        print(f"\n Successfully analyzed: {sampled_count} images")
        if failed_count > 0:
            print(f" Failed to read: {failed_count} images")
        
        # Convert to DataFrame
        prop_df = pd.DataFrame(properties)
        
        print("\nImage Properties Summary:")
        print(prop_df[['heights', 'widths', 'aspects', 'file_sizes', 
                      'mean_brightness', 'std_brightness']].describe())
        
        # Save statistics
        prop_df.describe().to_csv(self.output_dir / 'image_properties_stats.csv')
        
        # Visualizations
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # Height distribution
        axes[0, 0].hist(properties['heights'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Height Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Height (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(properties['heights']), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(properties["heights"]):.0f}')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Width distribution
        axes[0, 1].hist(properties['widths'], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Image Width Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(properties['widths']), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(properties["widths"]):.0f}')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Aspect ratio
        axes[0, 2].hist(properties['aspects'], bins=50, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Aspect Ratio Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Aspect Ratio (W/H)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(properties['aspects']), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(properties["aspects"]):.2f}')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # File size
        axes[1, 0].hist(properties['file_sizes'], bins=50, color='wheat', edgecolor='black')
        axes[1, 0].set_title('File Size Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Size (KB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(alpha=0.3)
        
        # Mean brightness
        axes[1, 1].hist(properties['mean_brightness'], bins=50, color='plum', edgecolor='black')
        axes[1, 1].set_title('Mean Brightness Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Mean Brightness (0-255)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(alpha=0.3)
        
        # Std brightness
        axes[1, 2].hist(properties['std_brightness'], bins=50, color='lightsalmon', edgecolor='black')
        axes[1, 2].set_title('Brightness Std Dev Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Std Deviation')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(alpha=0.3)
        
        # Scatter: Width vs Height
        axes[2, 0].scatter(properties['widths'], properties['heights'], 
                          alpha=0.5, c='steelblue', edgecolors='black', s=10)
        axes[2, 0].set_title('Width vs Height', fontweight='bold')
        axes[2, 0].set_xlabel('Width (pixels)')
        axes[2, 0].set_ylabel('Height (pixels)')
        axes[2, 0].grid(alpha=0.3)
        
        # Scatter: File Size vs Dimensions
        total_pixels = np.array(properties['widths']) * np.array(properties['heights'])
        axes[2, 1].scatter(total_pixels, properties['file_sizes'], 
                          alpha=0.5, c='coral', edgecolors='black', s=10)
        axes[2, 1].set_title('File Size vs Total Pixels', fontweight='bold')
        axes[2, 1].set_xlabel('Total Pixels')
        axes[2, 1].set_ylabel('File Size (KB)')
        axes[2, 1].grid(alpha=0.3)
        
        # Box plot: Dimensions by category
        if len(prop_df) > 0:
            prop_df.boxplot(column='widths', by='category', ax=axes[2, 2])
            axes[2, 2].set_title('Image Width by Category', fontweight='bold')
            axes[2, 2].set_xlabel('Category')
            axes[2, 2].set_ylabel('Width (pixels)')
            plt.sca(axes[2, 2])
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_properties.png', dpi=300, bbox_inches='tight')
        print(f" Saved: image_properties.png")
        plt.close()
        
    def visualize_sample_images(self, samples_per_class=5):
        """Visualize sample images from each class"""
        print("\n" + "=" * 80)
        print(f"CREATING SAMPLE IMAGE GRID ({samples_per_class} per class)")
        print("=" * 80)
        
        # Select subset of classes to visualize
        num_classes_to_show = min(15, len(self.class_info))
        
        # Sort by count and select top classes
        sorted_classes = sorted(self.class_info.items(), 
                               key=lambda x: x[1]['count'], reverse=True)
        selected_classes = sorted_classes[:num_classes_to_show]
        
        fig, axes = plt.subplots(num_classes_to_show, samples_per_class, 
                                figsize=(20, 3.5 * num_classes_to_show))
        
        if num_classes_to_show == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (class_name, info) in enumerate(selected_classes):
            images = info['images']
            samples = np.random.choice(images, min(samples_per_class, len(images)), replace=False)
            
            for j, img_path in enumerate(samples):
                try:
                    # Use safe image reading
                    img = self.read_image_safe(img_path)
                    
                    if img is not None:
                        axes[idx, j].imshow(img)
                except:
                    pass
                
                axes[idx, j].axis('off')
                
                if j == 0:
                    # Truncate long names
                    display_name = class_name if len(class_name) <= 40 else class_name[:37] + '...'
                    axes[idx, j].set_title(f"{display_name}\n({info['count']} images)", 
                                          fontsize=9, fontweight='bold', loc='left')
        
        plt.suptitle('Sample Images from Different Classes', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_images.png', dpi=200, bbox_inches='tight')
        print(f" Saved: sample_images.png")
        plt.close()
        
    def generate_summary_report(self, total_images):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        with open(self.output_dir / 'image_eda_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RICE DISEASE IMAGE DATASET - EDA SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Images: {total_images:,}\n")
            f.write(f"Total Classes: {len(self.class_info)}\n\n")
            
            # Count by category
            categories = {}
            for info in self.class_info.values():
                cat = info['category']
                categories[cat] = categories.get(cat, 0) + info['count']
            
            f.write("Images by Category:\n")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {cat}: {count:,} images\n")
            f.write("\n")
            
            f.write("2. CLASS DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            for class_name, info in sorted(self.class_info.items(), 
                                          key=lambda x: x[1]['count'], reverse=True):
                f.write(f"{info['count']:6,} images: {class_name} ({info['category']})\n")
            f.write("\n")
            
            f.write("3. DATA BALANCE ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            counts = [info['count'] for info in self.class_info.values()]
            f.write(f"Min images per class: {min(counts):,}\n")
            f.write(f"Max images per class: {max(counts):,}\n")
            f.write(f"Mean images per class: {np.mean(counts):.1f}\n")
            f.write(f"Median images per class: {np.median(counts):.1f}\n")
            f.write(f"Imbalance ratio (max/min): {max(counts)/min(counts):.2f}x\n\n")
            
            f.write("4. KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write(f"- Dataset contains {len(self.class_info)} different classes\n")
            f.write("- Classes are organized into 4 main categories:\n")
            f.write("  * Healthy rice plants\n")
            f.write("  * Rice diseases\n")
            f.write("  * Rice pests\n")
            f.write("  * Nutrient deficiencies\n")
            f.write("- Images vary in dimensions and quality\n")
            f.write("- Class imbalance present - need addressing during training\n\n")
            
            if len(self.read_errors) > 0:
                f.write(f"5. IMAGE READING ISSUES\n")
                f.write("-" * 80 + "\n")
                f.write(f"- {len(self.read_errors)} images could not be read\n")
                f.write("- Most likely due to corrupted files or unusual formats\n\n")
            
            f.write("6. RECOMMENDATIONS FOR MODELING\n")
            f.write("-" * 80 + "\n")
            f.write("- Resize all images to consistent dimensions (e.g., 224x224 or 299x299)\n")
            f.write("- Apply data augmentation for minority classes:\n")
            f.write("  * Random rotation, flip, brightness adjustment\n")
            f.write("  * Random crop, zoom, translation\n")
            f.write("- Use transfer learning (ResNet, EfficientNet, Vision Transformer)\n")
            f.write("- Implement stratified train-validation-test split\n")
            f.write("- Consider class weights or focal loss for imbalanced classes\n")
            f.write("- Use appropriate metrics: F1-score, precision, recall per class\n")
            f.write("- Consider hierarchical classification (category → specific disease)\n")
            f.write("- Use PIL-based data loaders to handle Unicode file paths\n")
            
        print(f" Saved: image_eda_summary.txt")
        
    def run_complete_analysis(self):
        """Run all analysis functions"""
        print("STARTING COMPREHENSIVE IMAGE EDA FOR RICE DISEASE DATASET")
        
        total_images = self.scan_directories()
        
        if total_images == 0:
            print("\n No images found! Please check your directory structure.")
            return
        
        self.analyze_class_distribution()
        self.analyze_image_properties()
        self.visualize_sample_images()
        self.generate_summary_report(total_images)
        
        if len(self.read_errors) > 0:
            print(f"\n Note: {len(self.read_errors)} images had reading issues")
            print("  Check the summary report for details.")
        
        print("IMAGE EDA COMPLETE! All outputs saved to:", self.output_dir)


if __name__ == "__main__":
    # Path to your dataset directory
    base_dir = r"C:\Users\admin\Downloads\DATASET"
    
    # Run analysis
    analyzer = RiceImageAnalyzer(base_dir)
    analyzer.run_complete_analysis()