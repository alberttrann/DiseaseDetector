"""
Smart hyperparameter tuning with manual grid search
Tests reasonable parameter ranges based on experience and EDA insights
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
import joblib

warnings.filterwarnings('ignore')

from npk_preprocessing import CropDataPreprocessor


class SmartHyperparamTuner:
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.results = {}
        
    def tune_random_forest(self):
        """Tune Random Forest with smart parameter grid"""
        print("\n" + "="*80)
        print("TUNING RANDOM FOREST")
        print("="*80)
        
        # Smart parameter grid based on dataset size and EDA
        param_grid = {
            'n_estimators': [100, 200, 300],           # More trees = better but slower
            'max_depth': [10, 15, 20, None],           # Depth control
            'min_samples_split': [2, 5, 10],           # Prevent overfitting
            'min_samples_leaf': [1, 2, 4],             # Leaf size
            'max_features': ['sqrt', 'log2', 0.5]      # Feature sampling
        }
        
        best_score = 0
        best_params = {}
        results = []
        
        total_combinations = (len(param_grid['n_estimators']) * 
                            len(param_grid['max_depth']) * 
                            len(param_grid['min_samples_split']) * 
                            len(param_grid['min_samples_leaf']) * 
                            len(param_grid['max_features']))
        
        print(f"\nTesting {total_combinations} parameter combinations...")
        print("This will test most impactful combinations only.\n")
        
        # Test only impactful combinations (reduce search space intelligently)
        pbar = tqdm(total=27)  # We'll test 27 smart combinations
        
        for n_est in [100, 200, 300]:
            for max_d in [15, 20, None]:
                for min_split in [2, 5, 10]:
                    # Skip unlikely combinations
                    if max_d == None and min_split == 10:
                        continue
                    
                    params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    
                    start_time = time.time()
                    
                    # Train and evaluate
                    rf = RandomForestClassifier(**params)
                    rf.fit(self.X_train, self.y_train)
                    
                    train_score = rf.score(self.X_train, self.y_train)
                    val_score = rf.score(self.X_val, self.y_val)
                    
                    elapsed = time.time() - start_time
                    
                    results.append({
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split,
                        'train_score': train_score,
                        'val_score': val_score,
                        'time': elapsed
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                    
                    pbar.update(1)
                    pbar.set_postfix({'best_val': f'{best_score*100:.2f}%'})
        
        pbar.close()
        
        print(f"\n{'='*80}")
        print("RANDOM FOREST BEST PARAMETERS:")
        print(f"{'='*80}")
        for param, value in best_params.items():
            if param not in ['random_state', 'n_jobs']:
                print(f"  {param}: {value}")
        print(f"\nBest Validation Score: {best_score*100:.2f}%")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_score', ascending=False)
        results_df.to_csv('models/tuning_rf_results.csv', index=False)
        
        self.results['random_forest'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }
        
        return best_params, best_score
    
    def tune_xgboost(self):
        """Tune XGBoost with smart parameter grid"""
        print("\n" + "="*80)
        print("TUNING XGBOOST")
        print("="*80)
        
        # Smart parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        best_score = 0
        best_params = {}
        results = []
        
        print(f"\nTesting smart XGBoost combinations...\n")
        
        pbar = tqdm(total=27)
        
        for n_est in [100, 200, 300]:
            for max_d in [6, 8, 10]:
                for lr in [0.01, 0.1, 0.3]:
                    params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'learning_rate': lr,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbosity': 0
                    }
                    
                    start_time = time.time()
                    
                    xgb = XGBClassifier(**params)
                    xgb.fit(self.X_train, self.y_train)
                    
                    train_score = xgb.score(self.X_train, self.y_train)
                    val_score = xgb.score(self.X_val, self.y_val)
                    
                    elapsed = time.time() - start_time
                    
                    results.append({
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'learning_rate': lr,
                        'train_score': train_score,
                        'val_score': val_score,
                        'time': elapsed
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                    
                    pbar.update(1)
                    pbar.set_postfix({'best_val': f'{best_score*100:.2f}%'})
        
        pbar.close()
        
        print(f"\n{'='*80}")
        print("XGBOOST BEST PARAMETERS:")
        print(f"{'='*80}")
        for param, value in best_params.items():
            if param not in ['random_state', 'n_jobs', 'verbosity']:
                print(f"  {param}: {value}")
        print(f"\nBest Validation Score: {best_score*100:.2f}%")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_score', ascending=False)
        results_df.to_csv('models/tuning_xgb_results.csv', index=False)
        
        self.results['xgboost'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }
        
        return best_params, best_score
    
    def tune_lightgbm(self):
        """Tune LightGBM with smart parameter grid"""
        print("\n" + "="*80)
        print("TUNING LIGHTGBM")
        print("="*80)
        
        best_score = 0
        best_params = {}
        results = []
        
        print(f"\nTesting smart LightGBM combinations...\n")
        
        pbar = tqdm(total=27)
        
        for n_est in [100, 200, 300]:
            for max_d in [6, 8, 10]:
                for num_leaves in [15, 31, 63]:
                    params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'num_leaves': num_leaves,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    
                    start_time = time.time()
                    
                    lgbm = LGBMClassifier(**params)
                    lgbm.fit(self.X_train, self.y_train)
                    
                    train_score = lgbm.score(self.X_train, self.y_train)
                    val_score = lgbm.score(self.X_val, self.y_val)
                    
                    elapsed = time.time() - start_time
                    
                    results.append({
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'num_leaves': num_leaves,
                        'train_score': train_score,
                        'val_score': val_score,
                        'time': elapsed
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                    
                    pbar.update(1)
                    pbar.set_postfix({'best_val': f'{best_score*100:.2f}%'})
        
        pbar.close()
        
        print(f"\n{'='*80}")
        print("LIGHTGBM BEST PARAMETERS:")
        print(f"{'='*80}")
        for param, value in best_params.items():
            if param not in ['random_state', 'n_jobs', 'verbose']:
                print(f"  {param}: {value}")
        print(f"\nBest Validation Score: {best_score*100:.2f}%")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_score', ascending=False)
        results_df.to_csv('models/tuning_lgbm_results.csv', index=False)
        
        self.results['lightgbm'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }
        
        return best_params, best_score
    
    def tune_catboost(self):
        """Tune CatBoost with smart parameter grid"""
        print("\n" + "="*80)
        print("TUNING CATBOOST")
        print("="*80)
        
        best_score = 0
        best_params = {}
        results = []
        
        print(f"\nTesting smart CatBoost combinations...\n")
        
        pbar = tqdm(total=27)
        
        for iterations in [100, 200, 300]:
            for depth in [6, 8, 10]:
                for lr in [0.01, 0.1, 0.3]:
                    params = {
                        'iterations': iterations,
                        'depth': depth,
                        'learning_rate': lr,
                        'random_state': 42,
                        'verbose': 0
                    }
                    
                    start_time = time.time()
                    
                    cat = CatBoostClassifier(**params)
                    cat.fit(self.X_train, self.y_train)
                    
                    train_score = cat.score(self.X_train, self.y_train)
                    val_score = cat.score(self.X_val, self.y_val)
                    
                    elapsed = time.time() - start_time
                    
                    results.append({
                        'iterations': iterations,
                        'depth': depth,
                        'learning_rate': lr,
                        'train_score': train_score,
                        'val_score': val_score,
                        'time': elapsed
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                    
                    pbar.update(1)
                    pbar.set_postfix({'best_val': f'{best_score*100:.2f}%'})
        
        pbar.close()
        
        print(f"\n{'='*80}")
        print("CATBOOST BEST PARAMETERS:")
        print(f"{'='*80}")
        for param, value in best_params.items():
            if param not in ['random_state', 'verbose']:
                print(f"  {param}: {value}")
        print(f"\nBest Validation Score: {best_score*100:.2f}%")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_score', ascending=False)
        results_df.to_csv('models/tuning_catboost_results.csv', index=False)
        
        self.results['catboost'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }
        
        return best_params, best_score
    
    def tune_all_models(self):
        """Tune all models"""
        print("STARTING HYPERPARAMETER TUNING")
        
        start_time = time.time()
        
        # Tune each model
        self.tune_random_forest()
        self.tune_xgboost()
        self.tune_lightgbm()
        self.tune_catboost()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("TUNING COMPLETE!")
        print("="*80)
        print(f"Total time: {total_time/60:.2f} minutes")
        
        # Summary
        self.print_summary()
        
        # Plot comparison
        self.plot_tuning_results()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all tuning results"""
        print("\n" + "="*80)
        print("TUNING SUMMARY")
        print("="*80)
        
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Best Val Score': f"{result['best_score']*100:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Best Val Score', ascending=False)
        print("\n", summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv('models/tuning_summary.csv', index=False)
        
        # Best overall
        best_model = max(self.results, key=lambda x: self.results[x]['best_score'])
        best_score = self.results[best_model]['best_score']
        
        print(f"\n{'='*80}")
        print(f" BEST MODEL: {best_model.upper()}")
        print(f" BEST SCORE: {best_score*100:.2f}%")
        print(f"{'='*80}")
    
    def plot_tuning_results(self):
        """Plot tuning results comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            df = result['all_results'].head(10)  # Top 10 results
            
            x = range(len(df))
            
            axes[idx].plot(x, df['train_score']*100, marker='o', label='Train', linewidth=2)
            axes[idx].plot(x, df['val_score']*100, marker='s', label='Val', linewidth=2)
            
            axes[idx].axhline(y=result['best_score']*100, color='r', linestyle='--', 
                            label=f"Best: {result['best_score']*100:.2f}%")
            
            axes[idx].set_xlabel('Configuration Rank', fontsize=11)
            axes[idx].set_ylabel('Accuracy (%)', fontsize=11)
            axes[idx].set_title(f'{model_name.upper()} Tuning Results', 
                              fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim([90, 101])
        
        plt.tight_layout()
        plt.savefig('models/tuning_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved tuning comparison plot: models/tuning_comparison.png")
        plt.close()
        
        # Plot parameter importance (for best model)
        self.plot_parameter_importance()
    
    def plot_parameter_importance(self):
        """Plot which parameters mattered most - FIXED VERSION"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            df = result['all_results']
            
            if model_name == 'random_forest':
                # Analyze n_estimators impact
                grouped = df.groupby('n_estimators')['val_score'].mean()
                axes[idx].bar(grouped.index.astype(str), grouped.values*100, color='skyblue')
                axes[idx].set_title(f'{model_name.upper()}: n_estimators Impact')
                axes[idx].set_ylabel('Avg Val Accuracy (%)')
                axes[idx].set_xlabel('n_estimators')
                
            elif model_name == 'xgboost':
                # Analyze learning_rate impact for XGBoost
                grouped = df.groupby('learning_rate')['val_score'].mean()
                axes[idx].bar(grouped.index.astype(str), grouped.values*100, color='lightcoral')
                axes[idx].set_title(f'{model_name.upper()}: Learning Rate Impact')
                axes[idx].set_ylabel('Avg Val Accuracy (%)')
                axes[idx].set_xlabel('learning_rate')
                
            elif model_name == 'lightgbm':
                # Analyze num_leaves impact for LightGBM (it has num_leaves, not learning_rate varied)
                grouped = df.groupby('num_leaves')['val_score'].mean()
                axes[idx].bar(grouped.index.astype(str), grouped.values*100, color='lightgreen')
                axes[idx].set_title(f'{model_name.upper()}: Num Leaves Impact')
                axes[idx].set_ylabel('Avg Val Accuracy (%)')
                axes[idx].set_xlabel('num_leaves')
                
            elif model_name == 'catboost':
                # Analyze depth impact
                grouped = df.groupby('depth')['val_score'].mean()
                axes[idx].bar(grouped.index.astype(str), grouped.values*100, color='wheat')
                axes[idx].set_title(f'{model_name.upper()}: Depth Impact')
                axes[idx].set_ylabel('Avg Val Accuracy (%)')
                axes[idx].set_xlabel('depth')
            
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_ylim([95, 101])  # Focus on the relevant range
        
        plt.tight_layout()
        plt.savefig('models/parameter_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved parameter importance plot: models/parameter_importance.png")
        plt.close()
    
    def save_best_params(self):
        """Save best parameters to file"""
        best_params_all = {}
        
        for model_name, result in self.results.items():
            best_params_all[model_name] = result['best_params']
        
        # Save as Python dict
        with open('models/best_hyperparameters.py', 'w') as f:
            f.write("# Best hyperparameters found through tuning\n\n")
            f.write("BEST_PARAMS = {\n")
            for model_name, params in best_params_all.items():
                f.write(f"    '{model_name}': {{\n")
                for param, value in params.items():
                    if isinstance(value, str):
                        f.write(f"        '{param}': '{value}',\n")
                    else:
                        f.write(f"        '{param}': {value},\n")
                f.write("    },\n")
            f.write("}\n")
        
        print("\n✓ Saved best hyperparameters to: models/best_hyperparameters.py")


if __name__ == "__main__":
    print(" Starting Smart Hyperparameter Tuning Pipeline")
    print("="*80)
    
    # Load data
    preprocessor = CropDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Tune
    tuner = SmartHyperparamTuner(X_train, X_val, y_train, y_val)
    results = tuner.tune_all_models()
    
    # Save best params
    tuner.save_best_params()
    
    print("TUNING PIPELINE COMPLETE!")
    print("\nNext steps:")
    print("1. Check 'models/tuning_summary.csv' for best scores")
    print("2. Review 'models/best_hyperparameters.py' for optimal parameters")
    print("3. Use these parameters to train final model on full dataset")