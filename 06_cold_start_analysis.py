#!/usr/bin/env python3
"""
06_cold_start_analysis.py
Cold-start risk modeling for new riders with minimal history
Generates performance metrics and risk score distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def identify_cold_start_orders(df):
    """Identify orders from riders in their first N orders"""
    print("="*80)
    print("COLD-START ANALYSIS")
    print("="*80)
    
    # Calculate order sequence for each rider
    df_sorted = df.sort_values(['rider_id', 'allot_time'])
    df_sorted['order_sequence'] = df_sorted.groupby('rider_id').cumcount() + 1
    
    # Define cold-start as first 10 orders
    cold_start_threshold = 10
    cold_start_orders = df_sorted[df_sorted['order_sequence'] <= cold_start_threshold].copy()
    
    print(f"\nCold-start definition: First {cold_start_threshold} orders")
    print(f"Cold-start orders: {len(cold_start_orders):,}")
    print(f"Unique riders: {cold_start_orders['rider_id'].nunique():,}")
    
    # Analyze cancellation patterns
    cold_cancel_rate = cold_start_orders['cancelled'].mean()
    warm_cancel_rate = df_sorted[df_sorted['order_sequence'] > cold_start_threshold]['cancelled'].mean()
    
    print(f"\nCancellation rates:")
    print(f"  Cold-start: {cold_cancel_rate:.2%}")
    print(f"  Warm-start: {warm_cancel_rate:.2%}")
    
    return cold_start_orders, df_sorted

def prepare_cold_start_features(cold_start_orders):
    """Prepare features for cold-start modeling"""
    
    # Focus on limited features available for new riders
    feature_cols = [
        'total_distance',
        'first_mile_distance', 
        'last_mile_distance',
        'hour',
        'is_peak_hour',
        'time_to_accept',
        'time_to_pickup',
        'order_sequence',  # How many orders they've done
        'session_time'
    ]
    
    # Create target: high-risk cancellation
    # For cold-start, we predict any cancellation (not just strategic)
    cold_start_orders['high_risk'] = (
        (cold_start_orders['cancelled'] == 1) & 
        (cold_start_orders['reason_text'].isin([
            'Cancel order due to bike issue',
            'Rider Cancelled - Couldn't find customer',
            'Cancelled by rider due to weather conditions'
        ]))
    ).astype(int)
    
    # Remove missing values
    cold_start_clean = cold_start_orders.dropna(subset=feature_cols + ['high_risk'])
    
    X = cold_start_clean[feature_cols]
    y = cold_start_clean['high_risk']
    
    print(f"\nCold-start modeling dataset:")
    print(f"  Total orders: {len(X):,}")
    print(f"  High-risk rate: {y.mean():.2%}")
    print(f"  Features: {len(feature_cols)}")
    
    return X, y, feature_cols, cold_start_clean

def train_cold_start_model(X, y):
    """Train cold-start risk model"""
    print("\n" + "="*50)
    print("TRAINING COLD-START MODEL")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=50,  # More conservative for cold-start
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    
    print(f"\nRandom Forest AUC: {auc_rf:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Low Risk', 'High Risk']))
    
    # Also train Logistic Regression for comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    lr_model.fit(X_train_scaled, y_train)
    
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba_lr)
    auc_lr = auc(fpr_lr, tpr_lr)
    
    print(f"\nLogistic Regression AUC: {auc_lr:.3f}")
    
    return {
        'rf': {'model': rf_model, 'fpr': fpr_rf, 'tpr': tpr_rf, 'auc': auc_rf, 
               'y_proba': y_proba_rf, 'thresholds': thresholds_rf},
        'lr': {'model': lr_model, 'fpr': fpr_lr, 'tpr': tpr_lr, 'auc': auc_lr,
               'y_proba': y_proba_lr, 'thresholds': thresholds_lr, 'scaler': scaler}
    }, X_test, y_test

def plot_cold_start_performance(models, X_test, y_test):
    """Plot cold-start model performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ROC curves
    ax1 = axes[0, 0]
    for name, data in models.items():
        label = 'Random Forest' if name == 'rf' else 'Logistic Regression'
        ax1.plot(data['fpr'], data['tpr'], linewidth=2, 
                label=f"{label} (AUC = {data['auc']:.3f})")
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - Cold-Start Models', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk score distribution
    ax2 = axes[0, 1]
    rf_scores = models['rf']['y_proba']
    
    # Create risk categories
    low_risk = rf_scores[y_test == 0]
    high_risk = rf_scores[y_test == 1]
    
    ax2.hist(low_risk, bins=30, alpha=0.6, label='Low Risk', color='green', density=True)
    ax2.hist(high_risk, bins=30, alpha=0.6, label='High Risk', color='red', density=True)
    ax2.axvline(x=0.3, color='black', linestyle='--', alpha=0.7)
    ax2.text(0.32, ax2.get_ylim()[1]*0.8, 'Threshold', fontsize=10)
    ax2.set_xlabel('Risk Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Performance by order sequence
    ax3 = axes[1, 0]
    
    # Get predictions by order sequence
    X_test_with_seq = X_test.copy()
    X_test_with_seq['y_true'] = y_test
    X_test_with_seq['risk_score'] = rf_scores
    
    seq_performance = []
    for seq in range(1, 11):
        seq_data = X_test_with_seq[X_test_with_seq['order_sequence'] == seq]
        if len(seq_data) > 10:  # Minimum sample size
            seq_performance.append({
                'sequence': seq,
                'high_risk_rate': seq_data['y_true'].mean(),
                'avg_risk_score': seq_data['risk_score'].mean()
            })
    
    if seq_performance:
        seq_df = pd.DataFrame(seq_performance)
        ax3.plot(seq_df['sequence'], seq_df['high_risk_rate'], 'o-', 
                label='Actual High Risk Rate', linewidth=2, markersize=8)
        ax3.plot(seq_df['sequence'], seq_df['avg_risk_score'], 's-', 
                label='Average Risk Score', linewidth=2, markersize=8)
        ax3.set_xlabel('Order Sequence', fontsize=12)
        ax3.set_ylabel('Rate / Score', fontsize=12)
        ax3.set_title('Risk by Order Sequence', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature importance
    ax4 = axes[1, 1]
    
    rf_model = models['rf']['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:8]  # Top 8 features
    
    feature_names = X_test.columns[indices]
    feature_imps = importances[indices]
    
    y_pos = np.arange(len(feature_names))
    ax4.barh(y_pos, feature_imps, color='skyblue', edgecolor='black', linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f.replace('_', ' ').title() for f in feature_names])
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title('Feature Importance - Cold-Start Model', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/cold_start_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Generated figures/cold_start_analysis.png")

def generate_cold_start_tables(models, X_test, y_test):
    """Generate LaTeX tables for cold-start analysis"""
    
    # Performance comparison table
    perf_table = """
\\begin{table}[H]
\\centering
\\caption{Cold-Start Model Performance Comparison}
\\label{tab:cold_start_performance}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{AUC} & \\textbf{Precision@30\\%} & \\textbf{Recall@30\\%} & \\textbf{F1@30\\%} \\\\
\\midrule
"""
    
    for name, data in models.items():
        model_name = 'Random Forest' if name == 'rf' else 'Logistic Regression'
        
        # Calculate metrics at 30% threshold
        threshold_idx = np.argmin(np.abs(data['thresholds'] - 0.3))
        threshold = data['thresholds'][threshold_idx]
        
        y_pred_at_threshold = (data['y_proba'] >= threshold).astype(int)
        
        tp = ((y_test == 1) & (y_pred_at_threshold == 1)).sum()
        fp = ((y_test == 0) & (y_pred_at_threshold == 1)).sum()
        fn = ((y_test == 1) & (y_pred_at_threshold == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        perf_table += f"{model_name} & {data['auc']:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} \\\\\n"
    
    perf_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_cold_start_performance.tex', 'w') as f:
        f.write(perf_table)
    
    print("✓ Generated tables/table_cold_start_performance.tex")
    
    # Risk categorization table
    rf_scores = models['rf']['y_proba']
    
    risk_categories = pd.cut(rf_scores, bins=[0, 0.2, 0.5, 1.0], 
                            labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    risk_table = """
\\begin{table}[H]
\\centering
\\caption{Cold-Start Risk Categorization}
\\label{tab:cold_start_risk_categories}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Risk Category} & \\textbf{Score Range} & \\textbf{Count} & \\textbf{Actual High Risk Rate} \\\\
\\midrule
"""
    
    for category in ['Low Risk', 'Medium Risk', 'High Risk']:
        mask = risk_categories == category
        count = mask.sum()
        actual_rate = y_test[mask].mean() if count > 0 else 0
        
        if category == 'Low Risk':
            score_range = '0.0 - 0.2'
        elif category == 'Medium Risk':
            score_range = '0.2 - 0.5'
        else:
            score_range = '0.5 - 1.0'
        
        risk_table += f"{category} & {score_range} & {count:,} & {actual_rate:.1%} \\\\\n"
    
    risk_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_cold_start_risk_categories.tex', 'w') as f:
        f.write(risk_table)
    
    print("✓ Generated tables/table_cold_start_risk_categories.tex")

def create_example_predictions(cold_start_clean, models, feature_cols):
    """Create example predictions for documentation"""
    
    # Get a few interesting examples
    examples = []
    
    # High risk example
    high_risk = cold_start_clean[
        (cold_start_clean['high_risk'] == 1) & 
        (cold_start_clean['order_sequence'] <= 3)
    ].head(1)
    
    # Low risk example  
    low_risk = cold_start_clean[
        (cold_start_clean['high_risk'] == 0) & 
        (cold_start_clean['order_sequence'] <= 3)
    ].head(1)
    
    # Medium risk example (predicted)
    rf_model = models['rf']['model']
    X_all = cold_start_clean[feature_cols]
    predictions = rf_model.predict_proba(X_all)[:, 1]
    
    medium_mask = (predictions > 0.3) & (predictions < 0.7)
    medium_risk = cold_start_clean[medium_mask].head(1)
    
    # Generate table
    example_table = """
\\begin{table}[H]
\\centering
\\caption{Example Cold-Start Risk Predictions}
\\label{tab:cold_start_examples}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Example} & \\textbf{Order \\#} & \\textbf{Distance} & \\textbf{Peak Hour} & \\textbf{Risk Score} & \\textbf{Outcome} \\\\
\\midrule
"""
    
    for idx, (label, df_ex) in enumerate([('High Risk', high_risk), 
                                          ('Low Risk', low_risk), 
                                          ('Medium Risk', medium_risk)]):
        if len(df_ex) > 0:
            row = df_ex.iloc[0]
            X_ex = df_ex[feature_cols]
            risk_score = rf_model.predict_proba(X_ex)[0, 1]
            
            outcome = 'Cancelled' if row.get('high_risk', 0) == 1 else 'Completed'
            
            example_table += f"{label} & {int(row['order_sequence'])} & "
            example_table += f"{row['total_distance']:.1f} km & "
            example_table += f"{'Yes' if row['is_peak_hour'] else 'No'} & "
            example_table += f"{risk_score:.3f} & {outcome} \\\\\n"
    
    example_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_cold_start_examples.tex', 'w') as f:
        f.write(example_table)
    
    print("✓ Generated tables/table_cold_start_examples.tex")

def save_cold_start_model(models, feature_cols):
    """Save cold-start model for deployment"""
    
    # Save Random Forest model (better performance)
    model_data = {
        'model': models['rf']['model'],
        'feature_cols': feature_cols,
        'auc': models['rf']['auc'],
        'model_type': 'cold_start_risk'
    }
    
    joblib.dump(model_data, 'models/cold_start_model.pkl')
    print("\n✓ Saved models/cold_start_model.pkl")

def main():
    """Main execution function"""
    
    # Create output directories
    import os
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("="*80)
    print("COLD-START RISK MODELING")
    print("="*80)
    
    # Load data
    df = pd.read_csv('shadowfax_processed-data-final.csv')
    
    # Identify cold-start orders
    cold_start_orders, df_with_sequence = identify_cold_start_orders(df)
    
    # Prepare features
    X, y, feature_cols, cold_start_clean = prepare_cold_start_features(cold_start_orders)
    
    # Train models
    models, X_test, y_test = train_cold_start_model(X, y)
    
    # Generate outputs
    plot_cold_start_performance(models, X_test, y_test)
    generate_cold_start_tables(models, X_test, y_test)
    create_example_predictions(cold_start_clean, models, feature_cols)
    
    # Save model
    save_cold_start_model(models, feature_cols)
    
    print("\n" + "="*80)
    print("COLD-START ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  Tables:")
    print("    - tables/table_cold_start_performance.tex")
    print("    - tables/table_cold_start_risk_categories.tex")
    print("    - tables/table_cold_start_examples.tex")
    print("  Figures:")
    print("    - figures/cold_start_analysis.png")
    print("  Models:")
    print("    - models/cold_start_model.pkl")

if __name__ == "__main__":
    main()