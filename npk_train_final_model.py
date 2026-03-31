"""
Train final models using best hyperparameters and save them
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from npk_preprocessing import CropDataPreprocessor


BEST_PARAMS = {
    'catboost': {
        'iterations': 200,
        'depth': 8,
        'learning_rate': 0.01,
        'random_state': 42,
        'verbose': 0
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'num_leaves': 15,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
}


def train_and_save_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train all models with best params and save them"""
    print("\n" + "="*80)
    print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    results = {}
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # 1. CatBoost (best performer)
    print("\n" + "-"*80)
    print("Training CatBoost...")
    print("-"*80)
    catboost_model = CatBoostClassifier(**BEST_PARAMS['catboost'])
    catboost_model.fit(X_train, y_train)
    
    train_acc = catboost_model.score(X_train, y_train)
    val_acc = catboost_model.score(X_val, y_val)
    test_acc = catboost_model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    joblib.dump(catboost_model, 'models/best_model_catboost.pkl')
    print(" Saved: models/best_model_catboost.pkl")
    
    results['catboost'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
    
    # 2. Random Forest
    print("\n" + "-"*80)
    print("Training Random Forest...")
    print("-"*80)
    rf_model = RandomForestClassifier(**BEST_PARAMS['random_forest'])
    rf_model.fit(X_train, y_train)
    
    train_acc = rf_model.score(X_train, y_train)
    val_acc = rf_model.score(X_val, y_val)
    test_acc = rf_model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    joblib.dump(rf_model, 'models/best_model_random_forest.pkl')
    print(" Saved: models/best_model_random_forest.pkl")
    
    results['random_forest'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
    
    # 3. XGBoost
    print("\n" + "-"*80)
    print("Training XGBoost...")
    print("-"*80)
    xgb_model = XGBClassifier(**BEST_PARAMS['xgboost'])
    xgb_model.fit(X_train, y_train)
    
    train_acc = xgb_model.score(X_train, y_train)
    val_acc = xgb_model.score(X_val, y_val)
    test_acc = xgb_model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    joblib.dump(xgb_model, 'models/best_model_xgboost.pkl')
    print(" Saved: models/best_model_xgboost.pkl")
    
    results['xgboost'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
    
    # 4. LightGBM
    print("\n" + "-"*80)
    print("Training LightGBM...")
    print("-"*80)
    lgbm_model = LGBMClassifier(**BEST_PARAMS['lightgbm'])
    lgbm_model.fit(X_train, y_train)
    
    train_acc = lgbm_model.score(X_train, y_train)
    val_acc = lgbm_model.score(X_val, y_val)
    test_acc = lgbm_model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    joblib.dump(lgbm_model, 'models/best_model_lightgbm.pkl')
    print(" Saved: models/best_model_lightgbm.pkl")
    
    results['lightgbm'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
    
    return results


def print_final_summary(results):
    """Print summary of all trained models"""
    print("\n" + "="*80)
    print("FINAL MODEL SUMMARY")
    print("="*80)
    
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Train Acc': f"{metrics['train']*100:.2f}%",
            'Val Acc': f"{metrics['val']*100:.2f}%",
            'Test Acc': f"{metrics['test']*100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test Acc', ascending=False)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('models/final_models_summary.csv', index=False)
    print("\n✓ Saved summary to: models/final_models_summary.csv")
    
    # Best model
    best_model = max(results, key=lambda x: results[x]['test'])
    best_test_acc = results[best_model]['test']
    
    print(f"\n{'='*80}")
    print(f" BEST MODEL: {best_model.upper()}")
    print(f" TEST ACCURACY: {best_test_acc*100:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    print(" Training Final Models with Best Hyperparameters")
    print("="*80)
    
    # Load preprocessed data
    preprocessor = CropDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()
    
    # Train and save all models
    results = train_and_save_all_models(
        X_train, X_val, X_test, 
        y_train, y_val, y_test
    )
    
    # Print summary
    print_final_summary(results)
    
    print("ALL MODELS TRAINED AND SAVED!")
    print("\nSaved models:")
    print("  - models/best_model_catboost.pkl")
    print("  - models/best_model_random_forest.pkl")
    print("  - models/best_model_xgboost.pkl")
    print("  - models/best_model_lightgbm.pkl")
    print("\nNow you can run: python npk_evaluate.py")