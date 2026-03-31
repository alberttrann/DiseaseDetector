"""
Comprehensive model evaluation on test set
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import pandas as pd
import json

from data_loaders import create_data_loaders
from models import create_model


class ModelEvaluator:
    def __init__(self, model, test_loader, classes, device='cuda', output_dir='evaluation_results'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.classes = classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_checkpoint(self, checkpoint_path):
        """Load trained model weights"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from epoch {checkpoint['epoch']} with val acc: {checkpoint['val_acc']:.2f}%")
        
    def predict(self):
        """Get predictions on test set"""
        print("\n" + "=" * 80)
        print("RUNNING PREDICTIONS ON TEST SET")
        print("=" * 80)
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate comprehensive metrics"""
        print("\n" + "=" * 80)
        print("CALCULATING METRICS")
        print("=" * 80)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.classes))
        )
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'Class': self.classes,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Sort by F1-score
        metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
        
        print("\nPer-Class Metrics:")
        print(metrics_df.to_string(index=False))
        
        # Save to CSV
        metrics_df.to_csv(self.output_dir / 'per_class_metrics.csv', index=False)
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\nMacro Average:")
        print(f"  Precision: {precision_macro*100:.2f}%")
        print(f"  Recall: {recall_macro*100:.2f}%")
        print(f"  F1-Score: {f1_macro*100:.2f}%")
        
        print(f"\nWeighted Average:")
        print(f"  Precision: {precision_weighted*100:.2f}%")
        print(f"  Recall: {recall_weighted*100:.2f}%")
        print(f"  F1-Score: {f1_weighted*100:.2f}%")
        
        # Save summary metrics
        summary = {
            'overall_accuracy': float(accuracy),
            'macro_precision': float(precision_macro),
            'macro_recall': float(recall_macro),
            'macro_f1': float(f1_macro),
            'weighted_precision': float(precision_weighted),
            'weighted_recall': float(recall_weighted),
            'weighted_f1': float(f1_weighted)
        }
        
        with open(self.output_dir / 'summary_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return metrics_df, summary
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        print("\n" + "=" * 80)
        print("GENERATING CONFUSION MATRIX")
        print("=" * 80)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Plot raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=self.classes, yticklabels=self.classes,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to: {self.output_dir / 'confusion_matrix.png'}")
        plt.close()
        
        # Save confusion matrix data
        cm_df = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        cm_df.to_csv(self.output_dir / 'confusion_matrix.csv')
    
    def plot_per_class_performance(self, metrics_df):
        """Plot per-class performance bars"""
        print("\n" + "=" * 80)
        print("GENERATING PER-CLASS PERFORMANCE CHARTS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Sort by F1-score for better visualization
        metrics_sorted = metrics_df.sort_values('F1-Score', ascending=True)
        
        # Plot 1: F1-Score
        axes[0, 0].barh(range(len(metrics_sorted)), metrics_sorted['F1-Score'], color='skyblue')
        axes[0, 0].set_yticks(range(len(metrics_sorted)))
        axes[0, 0].set_yticklabels(metrics_sorted['Class'], fontsize=9)
        axes[0, 0].set_xlabel('F1-Score', fontsize=11)
        axes[0, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        axes[0, 0].set_xlim([0, 1])
        
        # Plot 2: Precision
        axes[0, 1].barh(range(len(metrics_sorted)), metrics_sorted['Precision'], color='lightcoral')
        axes[0, 1].set_yticks(range(len(metrics_sorted)))
        axes[0, 1].set_yticklabels(metrics_sorted['Class'], fontsize=9)
        axes[0, 1].set_xlabel('Precision', fontsize=11)
        axes[0, 1].set_title('Precision by Class', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        axes[0, 1].set_xlim([0, 1])
        
        # Plot 3: Recall
        axes[1, 0].barh(range(len(metrics_sorted)), metrics_sorted['Recall'], color='lightgreen')
        axes[1, 0].set_yticks(range(len(metrics_sorted)))
        axes[1, 0].set_yticklabels(metrics_sorted['Class'], fontsize=9)
        axes[1, 0].set_xlabel('Recall', fontsize=11)
        axes[1, 0].set_title('Recall by Class', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        axes[1, 0].set_xlim([0, 1])
        
        # Plot 4: Support (sample count)
        axes[1, 1].barh(range(len(metrics_sorted)), metrics_sorted['Support'], color='wheat')
        axes[1, 1].set_yticks(range(len(metrics_sorted)))
        axes[1, 1].set_yticklabels(metrics_sorted['Class'], fontsize=9)
        axes[1, 1].set_xlabel('Number of Samples', fontsize=11)
        axes[1, 1].set_title('Test Set Support by Class', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        print(f" Saved per-class performance to: {self.output_dir / 'per_class_performance.png'}")
        plt.close()
    
    def analyze_misclassifications(self, y_true, y_pred):
        """Analyze most common misclassifications"""
        print("\n" + "=" * 80)
        print("ANALYZING MISCLASSIFICATIONS")
        print("=" * 80)
        
        # Get misclassified indices
        misclassified_idx = np.where(y_true != y_pred)[0]
        print(f"\nTotal misclassified: {len(misclassified_idx)} / {len(y_true)} ({len(misclassified_idx)/len(y_true)*100:.2f}%)")
        
        if len(misclassified_idx) == 0:
            print("Perfect accuracy! No misclassifications.")
            return
        
        # Count misclassification pairs
        misclass_pairs = {}
        for idx in misclassified_idx:
            true_class = self.classes[y_true[idx]]
            pred_class = self.classes[y_pred[idx]]
            pair = f"{true_class} → {pred_class}"
            misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 20 Misclassification Patterns:")
        print("-" * 80)
        for i, (pair, count) in enumerate(sorted_pairs[:20], 1):
            print(f"{i:2d}. {pair:60s} : {count:4d} times")
        
        # Save to file
        with open(self.output_dir / 'misclassifications.txt', 'w', encoding='utf-8') as f:
            f.write("MISCLASSIFICATION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total misclassified: {len(misclassified_idx)} / {len(y_true)}\n")
            f.write(f"Error rate: {len(misclassified_idx)/len(y_true)*100:.2f}%\n\n")
            f.write("All Misclassification Patterns:\n")
            f.write("-" * 80 + "\n")
            for pair, count in sorted_pairs:
                f.write(f"{pair:60s} : {count:4d} times\n")
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate sklearn classification report"""
        report = classification_report(
            y_true, y_pred, 
            target_names=self.classes, 
            digits=4
        )
        
        print("\n" + "=" * 80)
        print("SKLEARN CLASSIFICATION REPORT")
        print("=" * 80)
        print(report)
        
        # Save to file
        with open(self.output_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
    
    def evaluate_full(self, checkpoint_path):
        """Run complete evaluation pipeline"""
        print("STARTING COMPREHENSIVE MODEL EVALUATION")
        
        # Load model
        self.load_checkpoint(checkpoint_path)
        
        # Get predictions
        y_pred, y_true, y_probs = self.predict()
        
        # Calculate metrics
        metrics_df, summary = self.calculate_metrics(y_true, y_pred, y_probs)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_per_class_performance(metrics_df)
        
        # Analyze errors
        self.analyze_misclassifications(y_true, y_pred)
        
        # Classification report
        self.generate_classification_report(y_true, y_pred)
        
        print(f"EVALUATION COMPLETE! Results saved to: {self.output_dir}")
        
        return summary


if __name__ == "__main__":
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    _, _, test_loader, classes = create_data_loaders(
        batch_size=32, num_workers=4
    )
    
    # Create model
    model = create_model('efficientnet_b0', num_classes=len(classes))
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, classes, device=device)
    summary = evaluator.evaluate_full('checkpoints/best_model.pth')
    
    print("\nFinal Summary:")
    print(json.dumps(summary, indent=2))