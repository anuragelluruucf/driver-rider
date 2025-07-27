#!/usr/bin/env python3
"""
05_model_comparison.py
Predictive modeling with multiple algorithms and comprehensive validation
Generates model comparison tables, ROC curves, and confusion matrices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                           precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def prepare_modeling_data(df):
    """Prepare data for modeling"""
    print("="*80)
    print("PREPARING DATA FOR MODELING")
    print("="*80)
    
    # Focus on bike issue cancellations only
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue'].copy()
    
    # Create binary target: strategic (1) vs genuine (0)
    bike_issues['strategic'] = (bike_issues['cancel_after_pickup'] == 1).astype(int)
    
    # Feature columns
    feature_cols = [
        'total_distance', 'first_mile_distance', 'last_mile_distance',
        'time_to_accept', 'time_to_pickup', 'session_time',
        'hour', 'is_peak_hour', 'lifetime_order_count',
        'cancel_rate', 'bike_issue_rate', 'cancel_after_pickup_ratio'
    ]
    
    # Remove rows with missing features
    bike_issues_clean = bike_issues.dropna(subset=feature_cols + ['strategic'])
    
    X = bike_issues_clean[feature_cols]
    y = bike_issues_clean['strategic']
    
    print(f"\nDataset size: {len(X):,} bike issue cancellations")
    print(f"Strategic rate: {y.mean():.1%}")
    print(f"Features: {len(feature_cols)}")
    
    return X, y, feature_cols

def train_multiple_models(X, y):
    """Train multiple models and compare performance"""
    print("\n" + "="*50)
    print("TRAINING MULTIPLE MODELS")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20,
            random_state=42, n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=20, random_state=42
        ),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for models that benefit from it
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled if name in ['Logistic Regression', 'SVM'] 
                                   else X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'scaler': scaler if name in ['Logistic Regression', 'SVM'] else None
        }
        
        print(f"  AUC: {roc_auc:.3f}, Precision: {results[name]['precision']:.3f}, Recall: {results[name]['recall']:.3f}")
    
    return results, X_test, y_test

def create_model_comparison_table(results):
    """Generate LaTeX table comparing all models"""
    
    table = """
\\begin{table}[H]
\\centering
\\caption{Model Performance Comparison}
\\label{tab:model_comparison}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Model} & \\textbf{AUC} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{CV AUC} & \\textbf{CV Std} \\\\
\\midrule
"""
    
    # Sort by AUC
    sorted_models = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    
    for name, metrics in sorted_models:
        table += f"{name} & {metrics['roc_auc']:.3f} & {metrics['precision']:.3f} & "
        table += f"{metrics['recall']:.3f} & {metrics['f1']:.3f} & "
        table += f"{metrics['cv_mean']:.3f} & {metrics['cv_std']:.3f} \\\\\n"
    
    table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_model_comparison.tex', 'w') as f:
        f.write(table)
    
    print("\n✓ Generated tables/table_model_comparison.tex")

def plot_roc_curves(results):
    """Plot ROC curves for all models"""
    
    plt.figure(figsize=(10, 8))
    
    # Colors for different models
    colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange', 'purple']
    
    for (name, metrics), color in zip(results.items(), colors):
        plt.plot(metrics['fpr'], metrics['tpr'], linewidth=2, 
                label=f"{name} (AUC = {metrics['roc_auc']:.3f})", color=color)
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated figures/roc_curves_comparison.png")

def plot_precision_recall_curves(results):
    """Plot Precision-Recall curves for all models"""
    
    plt.figure(figsize=(10, 8))
    
    colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange', 'purple']
    
    for (name, metrics), color in zip(results.items(), colors):
        plt.plot(metrics['recall_curve'], metrics['precision_curve'], linewidth=2,
                label=f"{name} (AP = {metrics['pr_auc']:.3f})", color=color)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('figures/pr_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated figures/pr_curves_comparison.png")

def create_confusion_matrices(results):
    """Create confusion matrix plots for top models"""
    
    # Select top 3 models by AUC
    top_models = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, metrics) in enumerate(top_models):
        cm = metrics['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        ax = axes[idx]
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                   xticklabels=['Genuine', 'Strategic'],
                   yticklabels=['Genuine', 'Strategic'],
                   ax=ax, cbar_kws={'label': 'Proportion'})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{name}\n(AUC={metrics["roc_auc"]:.3f})', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrices_top3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated figures/confusion_matrices_top3.png")

def generate_detailed_metrics_table(results, model_name='Random Forest'):
    """Generate detailed metrics table for the best model"""
    
    metrics = results[model_name]
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Detailed Performance Metrics - {model_name}}}
\\label{{tab:detailed_metrics}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
True Positives & {tp:,} \\\\
True Negatives & {tn:,} \\\\
False Positives & {fp:,} \\\\
False Negatives & {fn:,} \\\\
\\midrule
Accuracy & {metrics['accuracy']:.3f} \\\\
Precision (PPV) & {metrics['precision']:.3f} \\\\
Recall (Sensitivity) & {metrics['recall']:.3f} \\\\
Specificity & {specificity:.3f} \\\\
F1-Score & {metrics['f1']:.3f} \\\\
\\midrule
ROC AUC & {metrics['roc_auc']:.3f} \\\\
PR AUC & {metrics['pr_auc']:.3f} \\\\
\\midrule
Cross-Val AUC & {metrics['cv_mean']:.3f} $\\pm$ {metrics['cv_std']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open('tables/table_detailed_metrics.tex', 'w') as f:
        f.write(table)
    
    print(f"✓ Generated tables/table_detailed_metrics.tex")

def analyze_feature_importance(results, X, feature_cols):
    """Analyze feature importance for tree-based models"""
    
    # Get Random Forest model
    rf_model = results['Random Forest']['model']
    
    # Feature importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot top 10 features
    top_n = 10
    top_indices = indices[:top_n]
    top_features = [feature_cols[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    colors = plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(top_features)))
    
    plt.barh(y_pos, top_importances, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (feat, imp) in enumerate(zip(top_features, top_importances)):
        plt.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=10)
    
    plt.yticks(y_pos, [f.replace('_', ' ').title() for f in top_features])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 10 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/feature_importance_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated figures/feature_importance_rf.png")
    
    # Generate LaTeX table
    importance_table = """
\\begin{table}[H]
\\centering
\\caption{Feature Importance Rankings}
\\label{tab:feature_importance}
\\begin{tabular}{lc}
\\toprule
\\textbf{Feature} & \\textbf{Importance} \\\\
\\midrule
"""
    
    for feat, imp in zip(top_features, top_importances):
        importance_table += f"{feat.replace('_', ' ').title()} & {imp:.3f} \\\\\n"
    
    importance_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_feature_importance.tex', 'w') as f:
        f.write(importance_table)
    
    print("✓ Generated tables/table_feature_importance.tex")

def save_best_model(results, X, y, feature_cols):
    """Save the best model for deployment"""
    
    # Get best model (highest AUC)
    best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_model = results[best_model_name]['model']
    scaler = results[best_model_name].get('scaler')
    
    print(f"\nSaving best model: {best_model_name}")
    
    # Save model
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': best_model_name,
        'metrics': {
            'auc': results[best_model_name]['roc_auc'],
            'precision': results[best_model_name]['precision'],
            'recall': results[best_model_name]['recall'],
            'f1': results[best_model_name]['f1']
        }
    }
    
    joblib.dump(model_data, 'models/best_strategic_model.pkl')
    print("✓ Saved models/best_strategic_model.pkl")
    
    # Also save as specific model type
    joblib.dump(model_data, f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl')
    print(f"✓ Saved models/{best_model_name.lower().replace(' ', '_')}_model.pkl")

def main():
    """Main execution function"""
    
    # Create output directories
    import os
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("="*80)
    print("MODEL COMPARISON AND VALIDATION")
    print("="*80)
    
    # Load labeled data
    try:
        df = pd.read_csv('data/labeled_orders.csv')
    except FileNotFoundError:
        print("Labeled data not found, using original dataset")
        df = pd.read_csv('shadowfax_processed-data-final.csv')
    
    # Prepare data
    X, y, feature_cols = prepare_modeling_data(df)
    
    # Train models
    results, X_test, y_test = train_multiple_models(X, y)
    
    # Generate outputs
    create_model_comparison_table(results)
    plot_roc_curves(results)
    plot_precision_recall_curves(results)
    create_confusion_matrices(results)
    generate_detailed_metrics_table(results)
    analyze_feature_importance(results, X, feature_cols)
    
    # Save best model
    save_best_model(results, X, y, feature_cols)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  Tables:")
    print("    - tables/table_model_comparison.tex")
    print("    - tables/table_detailed_metrics.tex")
    print("    - tables/table_feature_importance.tex")
    print("  Figures:")
    print("    - figures/roc_curves_comparison.png")
    print("    - figures/pr_curves_comparison.png")
    print("    - figures/confusion_matrices_top3.png")
    print("    - figures/feature_importance_rf.png")
    print("  Models:")
    print("    - models/best_strategic_model.pkl")

if __name__ == "__main__":
    main()