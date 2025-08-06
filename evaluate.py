"""
Evaluation and Plotting Module for Trading Predictor
==================================================

This module contains all evaluation metrics and plotting functions for the trading prediction system.
Implements comprehensive model evaluation, visualization, and performance analysis.

Features include:
- Comprehensive evaluation metrics for actionable trading
- Confusion matrix plotting
- Feature importance visualization
- Detailed classification reports
- Model performance comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional

# Scikit-learn imports for metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           balanced_accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve)

# Initialize logger
logger = logging.getLogger(__name__)

def evaluate_model_actionable(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation with actionability metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Prediction distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    total_pred = len(y_pred)
    
    metrics['pred_sell_pct'] = (pred_counts.get(0, 0) / total_pred) * 100
    metrics['pred_hold_pct'] = (pred_counts.get(1, 0) / total_pred) * 100
    metrics['pred_buy_pct'] = (pred_counts.get(2, 0) / total_pred) * 100
    
    # Actionability score
    metrics['actionability_score'] = (metrics['pred_buy_pct'] + metrics['pred_sell_pct']) / 100
    
    # Class-specific metrics
    unique_classes = np.unique(y_true)
    for cls in unique_classes:
        cls_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(cls, f'Class_{cls}')
        try:
            precision_cls = precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)
            recall_cls = recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)
            f1_cls = f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)
            
            metrics[f'precision_{cls_name}'] = precision_cls[0] if len(precision_cls) > 0 else 0
            metrics[f'recall_{cls_name}'] = recall_cls[0] if len(recall_cls) > 0 else 0
            metrics[f'f1_{cls_name}'] = f1_cls[0] if len(f1_cls) > 0 else 0
        except:
            metrics[f'precision_{cls_name}'] = 0
            metrics[f'recall_{cls_name}'] = 0
            metrics[f'f1_{cls_name}'] = 0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Generate and save confusion matrix plot comparing true labels vs predicted labels
    with class names SELL, HOLD, BUY.
    """
    print("📊 Generating confusion matrix...")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Define class names
    class_names = ['SELL', 'HOLD', 'BUY']
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    
    # Add labels and title
    plt.title('Confusion Matrix - Stock Action Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    
    # Add accuracy information
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
               fontsize=10, ha='left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved as '{save_path}'")
    
    return cm

def print_detailed_classification_report(y_true, y_pred):
    """
    Print detailed classification report showing precision, recall, and F1-score for each class.
    """
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print("=" * 60)
    
    # Define class names
    class_names = ['SELL', 'HOLD', 'BUY']
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names,
                                 digits=4, output_dict=True)
    
    # Print header
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    # Print metrics for each class
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            metrics = report[str(i)]
            print(f"{class_name:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")
    
    print("-" * 60)
    
    # Print overall metrics
    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"{'Macro Avg':<10} {macro['precision']:<12.4f} {macro['recall']:<12.4f} "
              f"{macro['f1-score']:<12.4f} {int(macro['support']):<10}")
    
    if 'weighted avg' in report:
        weighted = report['weighted avg']
        print(f"{'Weighted Avg':<10} {weighted['precision']:<12.4f} {weighted['recall']:<12.4f} "
              f"{weighted['f1-score']:<12.4f} {int(weighted['support']):<10}")
    
    # Print accuracy
    if 'accuracy' in report:
        accuracy = report['accuracy']
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return report

def plot_feature_importance(model, feature_names=None, top_n=20, save_path='feature_importance.png'):
    """
    Extract and plot feature importances from trained model.
    Assumes model has feature_importances_ attribute (tree-based models).
    """
    print("📊 Generating feature importance plot...")
    
    # Check if model has feature importances
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model does not have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Create DataFrame for easier handling
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = feature_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['importance'],
                   color='steelblue', alpha=0.7)
    
    # Customize plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - {type(model).__name__}',
             fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    # Invert y-axis to show highest importance at top
    plt.gca().invert_yaxis()
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Feature importance plot saved as '{save_path}'")
    
    # Print top features
    print(f"\n📊 Top {min(10, len(top_features))} Most Important Features:")
    print("-" * 50)
    for i, (_, row) in enumerate(top_features.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.6f}")
    
    return feature_df

def plot_model_comparison(results_df, save_path='model_comparison.png'):
    """
    Plot comparison of different models based on key metrics.
    """
    print("📊 Generating model comparison plot...")
    
    if results_df.empty:
        print("⚠️ No results to plot")
        return
    
    # Select key metrics for comparison
    key_metrics = []
    for prefix in ['test_', 'val_', '']:
        for metric in ['accuracy', 'f1_macro', 'actionability_score']:
            col_name = f"{prefix}{metric}" if prefix else metric
            if col_name in results_df.columns:
                key_metrics.append(col_name)
                break  # Use first available metric
    
    if not key_metrics:
        print("⚠️ No suitable metrics found for comparison")
        return
    
    # Create subplots for each metric
    n_metrics = len(key_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(key_metrics):
        # Sort models by metric value
        sorted_results = results_df.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        bars = axes[i].barh(range(len(sorted_results)), sorted_results[metric],
                           color='skyblue', alpha=0.7)
        
        # Customize plot
        axes[i].set_yticks(range(len(sorted_results)))
        axes[i].set_yticklabels(sorted_results.index, fontsize=10)
        axes[i].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            axes[i].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Add grid
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison plot saved as '{save_path}'")

def plot_prediction_distribution(y_true, y_pred, save_path='prediction_distribution.png'):
    """
    Plot distribution of predictions vs true labels.
    """
    print("📊 Generating prediction distribution plot...")
    
    # Define class names
    class_names = ['SELL', 'HOLD', 'BUY']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True label distribution
    true_counts = pd.Series(y_true).value_counts().sort_index()
    true_percentages = (true_counts / len(y_true)) * 100
    
    bars1 = ax1.bar(range(len(true_counts)), true_percentages, 
                   color=['red', 'gray', 'green'], alpha=0.7)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Predicted label distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    pred_percentages = (pred_counts / len(y_pred)) * 100
    
    bars2 = ax2.bar(range(len(pred_counts)), pred_percentages,
                   color=['red', 'gray', 'green'], alpha=0.7)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Label Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prediction distribution plot saved as '{save_path}'")

def generate_evaluation_summary(results_df, best_model_name=None, save_path='evaluation_summary.txt'):
    """
    Generate a comprehensive text summary of evaluation results.
    """
    print("📊 Generating evaluation summary...")
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("TRADING PREDICTOR EVALUATION SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    if not results_df.empty:
        summary_lines.append("MODEL PERFORMANCE RANKINGS:")
        summary_lines.append("-" * 40)
        
        # Rank models by combined score if available
        if 'final_combined_score' in results_df.columns:
            sorted_results = results_df.sort_values('final_combined_score', ascending=False)
            ranking_metric = 'final_combined_score'
        elif 'test_accuracy' in results_df.columns:
            sorted_results = results_df.sort_values('test_accuracy', ascending=False)
            ranking_metric = 'test_accuracy'
        else:
            sorted_results = results_df
            ranking_metric = 'N/A'
        
        for i, (model_name, row) in enumerate(sorted_results.head(10).iterrows()):
            summary_lines.append(f"{i+1:2d}. {model_name}")
            
            # Find best available metrics
            for metric_prefix in ['test_', 'val_', '']:
                accuracy_key = f"{metric_prefix}accuracy"
                f1_key = f"{metric_prefix}f1_macro"
                actionability_key = f"{metric_prefix}actionability_score"
                
                if accuracy_key in row and pd.notna(row[accuracy_key]):
                    summary_lines.append(f"    Accuracy: {row[accuracy_key]:.4f}")
                if f1_key in row and pd.notna(row[f1_key]):
                    summary_lines.append(f"    F1-Score: {row[f1_key]:.4f}")
                if actionability_key in row and pd.notna(row[actionability_key]):
                    summary_lines.append(f"    Actionability: {row[actionability_key]:.4f}")
                break
            
            summary_lines.append("")
        
        summary_lines.append("")
        
        if best_model_name:
            summary_lines.append(f"BEST MODEL: {best_model_name}")
            summary_lines.append("-" * 40)
            best_row = results_df.loc[best_model_name]
            
            # Add detailed metrics for best model
            for metric_prefix in ['test_', 'val_', '']:
                metrics_found = False
                for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'actionability_score']:
                    key = f"{metric_prefix}{metric}"
                    if key in best_row and pd.notna(best_row[key]):
                        summary_lines.append(f"{key.replace('_', ' ').title()}: {best_row[key]:.4f}")
                        metrics_found = True
                
                if metrics_found:
                    break
            
            summary_lines.append("")
    
    summary_lines.append("EVALUATION COMPLETED")
    summary_lines.append("=" * 80)
    
    # Save summary to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Print summary
    for line in summary_lines:
        print(line)
    
    print(f"\n✅ Evaluation summary saved to '{save_path}'")