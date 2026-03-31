"""
Predict optimal crop for given soil and climate conditions
Fixes: feature name warnings, validates all 29 features
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load saved model and preprocessor
print("Loading model and preprocessor...")
model = joblib.load('models/best_model_lightgbm.pkl')
scaler = joblib.load('models/preprocessor/scaler.pkl')
label_encoder = joblib.load('models/preprocessor/label_encoder.pkl')
feature_names = joblib.load('models/preprocessor/feature_names.pkl')

print(f" Model loaded")
print(f" Scaler loaded")
print(f" Label encoder loaded with {len(label_encoder.classes_)} crops")
print(f" Feature names loaded: {len(feature_names)} features expected")
print()

def engineer_features(input_data):
    """
    Apply same feature engineering as training
    Input: dict with keys: N, P, K, temperature, humidity, ph, rainfall
    Output: DataFrame with all 29 features
    """
    df = pd.DataFrame([input_data])
    
    # 1. NPK Ratios
    df['N_P_ratio'] = df['N'] / (df['P'] + 1e-6)
    df['N_K_ratio'] = df['N'] / (df['K'] + 1e-6)
    df['P_K_ratio'] = df['P'] / (df['K'] + 1e-6)
    
    # 2. Total nutrients
    df['total_NPK'] = df['N'] + df['P'] + df['K']
    
    # 3. Climate interactions
    df['temp_humidity_product'] = df['temperature'] * df['humidity']
    df['temp_humidity_diff'] = df['temperature'] - df['humidity']
    
    # 4. Moisture index
    df['moisture_index'] = df['rainfall'] * df['humidity'] / 100
    
    # 5. pH deviation from neutral
    df['ph_neutral_dev'] = np.abs(df['ph'] - 7.0)
    
    # 6. Polynomial features
    df['N_squared'] = df['N'] ** 2
    df['P_squared'] = df['P'] ** 2
    df['K_squared'] = df['K'] ** 2
    
    # 7. Interaction terms
    df['N_temp'] = df['N'] * df['temperature']
    df['P_rainfall'] = df['P'] * df['rainfall']
    df['K_humidity'] = df['K'] * df['humidity']
    
    # 8. Climate zones (binning) - CRITICAL: must match training exactly
    df['temp_zone'] = pd.cut(df['temperature'], 
                              bins=[0, 15, 25, 35, 50],
                              labels=['cold', 'moderate', 'warm', 'hot'])
    df['humidity_zone'] = pd.cut(df['humidity'],
                                 bins=[0, 40, 70, 100],
                                 labels=['dry', 'moderate', 'humid'])
    df['rainfall_zone'] = pd.cut(df['rainfall'],
                                 bins=[0, 50, 100, 200, 300],
                                 labels=['low', 'medium', 'high', 'very_high'])
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['temp_zone', 'humidity_zone', 'rainfall_zone'],
                        drop_first=True)
    
    # Ensure all expected columns exist (add missing with 0)
    missing_cols = [col for col in feature_names if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    
    # Reorder to match training exactly
    df = df[feature_names]
    
    # Validate: check we have exactly 29 features
    assert len(df.columns) == 29, f"Expected 29 features, got {len(df.columns)}"
    assert set(df.columns) == set(feature_names), "Feature mismatch!"
    
    return df


def predict_crop(N, P, K, temperature, humidity, ph, rainfall, verbose=False):
    """
    Predict optimal crop for given conditions
    
    Parameters:
    -----------
    N : float - Nitrogen content ratio
    P : float - Phosphorous content ratio
    K : float - Potassium content ratio
    temperature : float - Temperature in °C
    humidity : float - Relative humidity in %
    ph : float - pH value of soil (0-14)
    rainfall : float - Rainfall in mm
    verbose : bool - Print debug info
    
    Returns:
    --------
    dict with keys:
        - crop: predicted crop name (str)
        - confidence: probability of top prediction (0-1)
        - top_3: list of (crop, probability) tuples
    """
    if verbose:
        print(f"Input: N={N}, P={P}, K={K}, T={temperature}°C, H={humidity}%, pH={ph}, R={rainfall}mm")
    
    # Create input dict
    input_data = {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    
    # 1. Engineer features
    X = engineer_features(input_data)
    
    if verbose:
        print(f" Engineered {len(X.columns)} features")
        print(f"  Features: {list(X.columns)}")
    
    # 2. Scale features (DON'T convert to DataFrame yet - causes warning)
    X_scaled_array = scaler.transform(X)
    
    if verbose:
        print(f" Features scaled using RobustScaler")
    
    # 3. Create DataFrame with correct feature names for LightGBM
    # This is what LightGBM expects - a DataFrame with the exact column names it was trained on
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_names)
    
    # 4. Predict (pass DataFrame directly - LightGBM will validate feature names)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Decode label
    crop_name = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = [
        (label_encoder.inverse_transform([idx])[0], probabilities[idx])
        for idx in top_3_idx
    ]
    
    if verbose:
        print(f" Prediction complete")
    
    return {
        'crop': crop_name,
        'confidence': confidence,
        'top_3': top_3_crops
    }


if __name__ == "__main__":
    # Example 1: Rice conditions (from dataset)
    print("="*80)
    print("EXAMPLE 1: Rice-suitable conditions")
    print("="*80)
    result = predict_crop(
        N=90, P=42, K=43,
        temperature=21, humidity=82,
        ph=6.5, rainfall=203,
        verbose=True
    )
    print(f"\n Recommended Crop: {result['crop'].upper()}")
    print(f" Confidence: {result['confidence']*100:.2f}%")
    print(f"\nTop 3 recommendations:")
    for i, (crop, prob) in enumerate(result['top_3'], 1):
        print(f"  {i}. {crop}: {prob*100:.2f}%")
    
    # Example 2: Apple conditions
    print("\n" + "="*80)
    print("EXAMPLE 2: Apple-suitable conditions")
    print("="*80)
    result = predict_crop(
        N=20, P=130, K=200,
        temperature=22, humidity=92,
        ph=6.0, rainfall=110,
        verbose=True
    )
    print(f"\n Recommended Crop: {result['crop'].upper()}")
    print(f" Confidence: {result['confidence']*100:.2f}%")
    print(f"\nTop 3 recommendations:")
    for i, (crop, prob) in enumerate(result['top_3'], 1):
        print(f"  {i}. {crop}: {prob*100:.2f}%")
    
    # Example 3: Custom input
    print("\n" + "="*80)
    print("EXAMPLE 3: Your custom input")
    print("="*80)
    result = predict_crop(
        N=85, P=60, K=45,
        temperature=28, humidity=75,
        ph=6.8, rainfall=150,
        verbose=True
    )
    print(f"\n Recommended Crop: {result['crop'].upper()}")
    print(f" Confidence: {result['confidence']*100:.2f}%")
    print(f"\nTop 3 recommendations:")
    for i, (crop, prob) in enumerate(result['top_3'], 1):
        print(f"  {i}. {crop}: {prob*100:.2f}%")
    
    # Example 4: Batch test all expected inputs
    print("\n" + "="*80)
    print("VALIDATION: Testing with diverse inputs")
    print("="*80)
    test_cases = [
        ("Cold + Low Rainfall", 30, 20, 15, 8, 30, 5.0, 25),
        ("Hot + High Rainfall", 120, 80, 70, 38, 85, 8.0, 250),
        ("Moderate", 50, 50, 50, 25, 70, 6.5, 100),
    ]
    
    for name, N, P, K, temp, humid, pH, rain in test_cases:
        result = predict_crop(N, P, K, temp, humid, pH, rain, verbose=False)
        print(f"\n{name}:")
        print(f"  → {result['crop']} ({result['confidence']*100:.1f}%)")