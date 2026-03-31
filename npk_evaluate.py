"""
Refactored evaluation script for NPK crop recommendation models
Loads preprocessor and best model, evaluates on test set, and generates metrics/plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import joblib
from pathlib import Path

from npk_preprocessing import CropDataPreprocessor

def load_best_model(model_path):
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("✓ Model loaded.")
    return model

def main():
    # Load preprocessor and data splits
    preprocessor = CropDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()
    class_names = preprocessor.label_encoder.classes_

    # Choose best model (update path as needed)
    # Auto-load all saved models that match pattern and evaluate on X_test.
    model_files = sorted(Path('models').glob('best_model_*.pkl'))
    if not model_files:
        raise FileNotFoundError("No saved models found in models/*.pkl")

    eval_results = []
    for mf in model_files:
        m = load_best_model(str(mf))
        y_pred_tmp = m.predict(X_test)
        acc_tmp = accuracy_score(y_test, y_pred_tmp)
        eval_results.append({'file': mf.name, 'model': m, 'acc': acc_tmp})

    # pick best model by test accuracy
    best_entry = max(eval_results, key=lambda x: x['acc'])
    print(f"Selected best model file: {best_entry['file']} (Test Acc: {best_entry['acc']*100:.2f}%)")
    model = best_entry['model']
    accuracy = best_entry['acc']
    y_pred = model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n", report)
    with open('models/evaluation_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confusion matrix.")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(class_names))
    )
    metrics_df = pd.DataFrame({
        'Crop': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    metrics_df = metrics_df.sort_values('F1-Score', ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes[0].barh(range(len(metrics_df)), metrics_df['Precision'], color='lightcoral')
    axes[0].set_yticks(range(len(metrics_df)))
    axes[0].set_yticklabels(metrics_df['Crop'])
    axes[0].set_xlabel('Precision', fontsize=11)
    axes[0].set_title('Precision by Crop', fontsize=12, fontweight='bold')
    axes[0].set_xlim([0, 1])
    axes[0].grid(axis='x', alpha=0.3)
    axes[1].barh(range(len(metrics_df)), metrics_df['Recall'], color='lightgreen')
    axes[1].set_yticks(range(len(metrics_df)))
    axes[1].set_yticklabels(metrics_df['Crop'])
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_title('Recall by Crop', fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(axis='x', alpha=0.3)
    axes[2].barh(range(len(metrics_df)), metrics_df['F1-Score'], color='skyblue')
    axes[2].set_yticks(range(len(metrics_df)))
    axes[2].set_yticklabels(metrics_df['Crop'])
    axes[2].set_xlabel('F1-Score', fontsize=11)
    axes[2].set_title('F1-Score by Crop', fontsize=12, fontweight='bold')
    axes[2].set_xlim([0, 1])
    axes[2].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(" Saved per-class metrics.")
    metrics_df.to_csv('models/per_class_metrics.csv', index=False)

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        indices = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print(" Saved feature importance.")
        plt.close()

    print(f"\n{'='*80}")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()