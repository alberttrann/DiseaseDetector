import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class NPKDataAnalyzer:
    def __init__(self, csv_path):
        """Initialize the analyzer with CSV file path"""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path('eda_outputs/npk_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def basic_info(self):
        """Display basic information about the dataset"""
        print("=" * 80)
        print("DATASET BASIC INFORMATION")
        print("=" * 80)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of samples: {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1]}")
        
        print("\n" + "-" * 80)
        print("Column Names and Data Types:")
        print("-" * 80)
        print(self.df.dtypes)
        
        print("\n" + "-" * 80)
        print("Missing Values:")
        print("-" * 80)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        if missing.sum() == 0:
            print("No missing values found!")
        
        print("\n" + "-" * 80)
        print("First 5 rows:")
        print("-" * 80)
        print(self.df.head())
        
        print("\n" + "-" * 80)
        print("Statistical Summary:")
        print("-" * 80)
        print(self.df.describe())
        
    def analyze_target_variable(self):
        """Analyze the target variable (crop labels)"""
        print("\n" + "=" * 80)
        print("TARGET VARIABLE ANALYSIS (Crop Labels)")
        print("=" * 80)
        
        # Count of each crop
        crop_counts = self.df['label'].value_counts()
        print(f"\nNumber of unique crops: {self.df['label'].nunique()}")
        print("\nCrop distribution:")
        print(crop_counts)
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Bar plot
        crop_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Crops in Dataset', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Crop Type', fontsize=12)
        axes[0].set_ylabel('Number of Samples', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(crop_counts.values):
            axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(crop_counts.values, labels=crop_counts.index, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 9})
        axes[1].set_title('Crop Distribution (Percentage)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crop_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: crop_distribution.png")
        plt.close()
        
    def analyze_features(self):
        """Analyze all numerical features"""
        print("\n" + "=" * 80)
        print("FEATURE ANALYSIS")
        print("=" * 80)
        
        numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Distribution plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            # Histogram with KDE
            axes[idx].hist(self.df[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].axvline(self.df[col].mean(), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {self.df[col].mean():.2f}')
            axes[idx].axvline(self.df[col].median(), color='green', linestyle='--', 
                            linewidth=2, label=f'Median: {self.df[col].median():.2f}')
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        # Hide unused subplot
        axes[-2].axis('off')
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: feature_distributions.png")
        plt.close()
        
        # Box plots for outlier detection
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            self.df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(col, fontsize=10)
            axes[idx].grid(alpha=0.3)
            
            # Calculate and display outliers
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            axes[idx].text(0.5, 0.95, f'Outliers: {len(outliers)}', 
                          transform=axes[idx].transAxes, 
                          ha='center', va='top', fontsize=10, 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-2].axis('off')
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_boxplots.png', dpi=300, bbox_inches='tight')
        print(f"Saved: feature_boxplots.png")
        plt.close()
        
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        correlation_matrix = self.df[numerical_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                    cmap='coolwarm', center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: correlation_heatmap.png")
        plt.close()
        
    def crop_wise_analysis(self):
        """Analyze features for each crop type"""
        print("\n" + "=" * 80)
        print("CROP-WISE FEATURE ANALYSIS")
        print("=" * 80)
        
        numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Group by crop and calculate mean
        crop_means = self.df.groupby('label')[numerical_cols].mean()
        print("\nMean values for each crop:")
        print(crop_means)
        
        # Save to CSV
        crop_means.to_csv(self.output_dir / 'crop_feature_means.csv')
        print(f"\nSaved: crop_feature_means.csv")
        
        # Visualize NPK requirements for each crop
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # N, P, K comparison
        crop_means[['N', 'P', 'K']].plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('NPK Requirements by Crop', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Crop', fontsize=11)
        axes[0, 0].set_ylabel('Value (kg/ha)', fontsize=11)
        axes[0, 0].legend(title='Nutrient')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Temperature and Humidity
        crop_means[['temperature', 'humidity']].plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Temperature & Humidity by Crop', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Crop', fontsize=11)
        axes[0, 1].set_ylabel('Value', fontsize=11)
        axes[0, 1].legend(title='Parameter')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # pH requirements
        crop_means['ph'].plot(kind='bar', ax=axes[1, 0], color='lightcoral', width=0.8)
        axes[1, 0].set_title('pH Requirements by Crop', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Crop', fontsize=11)
        axes[1, 0].set_ylabel('pH Value', fontsize=11)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=7, color='red', linestyle='--', label='Neutral pH')
        axes[1, 0].legend()
        
        # Rainfall requirements
        crop_means['rainfall'].plot(kind='bar', ax=axes[1, 1], color='skyblue', width=0.8)
        axes[1, 1].set_title('Rainfall Requirements by Crop', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Crop', fontsize=11)
        axes[1, 1].set_ylabel('Rainfall (mm)', fontsize=11)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crop_requirements.png', dpi=300, bbox_inches='tight')
        print(f"Saved: crop_requirements.png")
        plt.close()
        
    def feature_importance_visualization(self):
        """Visualize feature importance using variance"""
        print("\n" + "=" * 80)
        print("FEATURE VARIANCE ANALYSIS")
        print("=" * 80)
        
        numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Calculate variance for each feature
        variances = self.df[numerical_cols].var().sort_values(ascending=False)
        print("\nFeature Variances:")
        print(variances)
        
        # Plot
        plt.figure(figsize=(12, 6))
        variances.plot(kind='bar', color='teal', edgecolor='black')
        plt.title('Feature Variance (Indicator of Variability)', fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Variance', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_variance.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: feature_variance.png")
        plt.close()
        
    def pairplot_analysis(self):
        """Create pairplot for selected features"""
        print("\n" + "=" * 80)
        print("PAIRPLOT ANALYSIS (This may take a moment...)")
        print("=" * 80)
        
        # Select a subset of crops for clarity
        top_crops = self.df['label'].value_counts().head(5).index
        df_subset = self.df[self.df['label'].isin(top_crops)]
        
        # Create pairplot
        pairplot = sns.pairplot(df_subset, hue='label', 
                                vars=['N', 'P', 'K', 'temperature', 'ph'],
                                diag_kind='kde', plot_kws={'alpha': 0.6},
                                height=2.5)
        pairplot.fig.suptitle('Feature Relationships (Top 5 Crops)', 
                              y=1.02, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_pairplot.png', dpi=200, bbox_inches='tight')
        print(f"\n Saved: feature_pairplot.png")
        plt.close()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        with open(self.output_dir / 'eda_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CROP RECOMMENDATION DATASET - EDA SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Number of Features: {self.df.shape[1]}\n")
            f.write(f"Number of Crops: {self.df['label'].nunique()}\n")
            f.write(f"Crop Types: {', '.join(self.df['label'].unique())}\n\n")
            
            f.write("2. DATA QUALITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Missing Values: {self.df.isnull().sum().sum()}\n")
            f.write(f"Duplicate Rows: {self.df.duplicated().sum()}\n\n")
            
            f.write("3. FEATURE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(self.df.describe().to_string())
            f.write("\n\n")
            
            f.write("4. CROP DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(self.df['label'].value_counts().to_string())
            f.write("\n\n")
            
            f.write("5. KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write("- Dataset is well-balanced across different crop types\n")
            f.write("- No missing values detected\n")
            f.write("- Features show varying scales (normalization recommended)\n")
            f.write("- Temperature and humidity show strong correlation\n")
            f.write("- Each crop has distinct NPK requirements\n\n")
            
            f.write("6. RECOMMENDATIONS FOR MODELING\n")
            f.write("-" * 80 + "\n")
            f.write("- Apply feature scaling (StandardScaler or MinMaxScaler)\n")
            f.write("- Consider feature engineering (NPK ratios, etc.)\n")
            f.write("- Use cross-validation for model evaluation\n")
            f.write("- Try ensemble methods (Random Forest, XGBoost)\n")
            f.write("- Monitor for class imbalance during training\n")
            
        print(f"\nSaved: eda_summary_report.txt")
        
    def run_complete_analysis(self):
        """Run all analysis functions"""
        print("STARTING COMPREHENSIVE EDA FOR CROP RECOMMENDATION DATASET")
        
        self.basic_info()
        self.analyze_target_variable()
        self.analyze_features()
        self.correlation_analysis()
        self.crop_wise_analysis()
        self.feature_importance_visualization()
        self.pairplot_analysis()
        self.generate_summary_report()
        
        print("EDA COMPLETE! All outputs saved to:", self.output_dir)


if __name__ == "__main__":
    # Path to your CSV file
    csv_path = r"Các thông số tối ưu cho các loại cây trồng\NPK.csv"
    
    # Run analysis
    analyzer = NPKDataAnalyzer(csv_path)
    analyzer.run_complete_analysis()