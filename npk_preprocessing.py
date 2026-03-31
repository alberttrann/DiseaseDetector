"""
Advanced preprocessing pipeline for NPK crop recommendation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

class CropDataPreprocessor:
    def __init__(self, data_path='Các thông số tối ưu cho các loại cây trồng/NPK.csv'):
        self.data_path = data_path
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_data(self):
        """Load and initial inspection"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        df = pd.read_csv(self.data_path)
        print(f"\nDataset shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")
        print(f"\nClass distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def feature_engineering(self, df):
        """Create advanced features based on EDA insights"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        df_eng = df.copy()
        
        # 1. NPK Ratios (important for nutrient balance)
        df_eng['N_P_ratio'] = df_eng['N'] / (df_eng['P'] + 1e-6)
        df_eng['N_K_ratio'] = df_eng['N'] / (df_eng['K'] + 1e-6)
        df_eng['P_K_ratio'] = df_eng['P'] / (df_eng['K'] + 1e-6)
        
        # 2. Total nutrients
        df_eng['total_NPK'] = df_eng['N'] + df_eng['P'] + df_eng['K']
        
        # 3. Climate comfort index (from EDA: temp and humidity are correlated)
        df_eng['temp_humidity_product'] = df_eng['temperature'] * df_eng['humidity']
        df_eng['temp_humidity_diff'] = df_eng['temperature'] - df_eng['humidity']
        
        # 4. Soil moisture indicator
        df_eng['moisture_index'] = df_eng['rainfall'] * df_eng['humidity'] / 100
        
        # 5. pH deviation from neutral
        df_eng['ph_neutral_dev'] = np.abs(df_eng['ph'] - 7.0)
        
        # 6. Polynomial features for N, P, K (capture non-linear relationships)
        df_eng['N_squared'] = df_eng['N'] ** 2
        df_eng['P_squared'] = df_eng['P'] ** 2
        df_eng['K_squared'] = df_eng['K'] ** 2
        
        # 7. Interaction terms (from correlation heatmap insights)
        df_eng['N_temp'] = df_eng['N'] * df_eng['temperature']
        df_eng['P_rainfall'] = df_eng['P'] * df_eng['rainfall']
        df_eng['K_humidity'] = df_eng['K'] * df_eng['humidity']
        
        # 8. Climate zones (binning)
        df_eng['temp_zone'] = pd.cut(df_eng['temperature'], 
                                      bins=[0, 15, 25, 35, 50],
                                      labels=['cold', 'moderate', 'warm', 'hot'])
        df_eng['humidity_zone'] = pd.cut(df_eng['humidity'],
                                         bins=[0, 40, 70, 100],
                                         labels=['dry', 'moderate', 'humid'])
        df_eng['rainfall_zone'] = pd.cut(df_eng['rainfall'],
                                         bins=[0, 50, 100, 200, 300],
                                         labels=['low', 'medium', 'high', 'very_high'])
        
        # One-hot encode categorical features
        df_eng = pd.get_dummies(df_eng, columns=['temp_zone', 'humidity_zone', 'rainfall_zone'],
                                drop_first=True)
        
        print(f"\n✓ Original features: 7")
        print(f"✓ Engineered features: {len(df_eng.columns) - 8}")  # -8 for original + label
        print(f"✓ Total features: {len(df_eng.columns) - 1}")  # -1 for label
        
        return df_eng
    
    def prepare_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Split and scale data"""
        print("\n" + "="*80)
        print("PREPARING TRAIN/VAL/TEST SPLITS")
        print("="*80)
        
        # Separate features and target
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            stratify=y_encoded, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            stratify=y_temp, random_state=random_state
        )
        
        print(f"\nTrain samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Val samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale features using RobustScaler (better for outliers based on EDA)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        print("\n✓ Features scaled using RobustScaler")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)
    
    def save_preprocessor(self, output_dir='models/preprocessor'):
        """Save scaler and encoder"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, output_dir / 'label_encoder.pkl')
        joblib.dump(self.feature_names, output_dir / 'feature_names.pkl')
        
        print(f"\n✓ Preprocessor saved to: {output_dir}")
    
    def run_pipeline(self):
        """Execute complete preprocessing pipeline"""
        # Load data
        df = self.load_data()
        
        # Feature engineering
        df_eng = self.feature_engineering(df)
        
        # Prepare splits
        splits = self.prepare_data(df_eng)
        
        # Save preprocessor
        self.save_preprocessor()
        
        return splits


if __name__ == "__main__":
    preprocessor = CropDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")