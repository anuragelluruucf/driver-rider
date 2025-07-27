#!/usr/bin/env python3
"""
08_robustness_checks.py
Robustness checks and sensitivity analysis
Tests model stability across different conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def test_label_sensitivity(df):
    """Test sensitivity to labeling threshold changes"""
    print("="*80)
    print("LABEL SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Prepare base features
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue'].copy()
    
    feature_cols = [
        'total_distance', 'first_mile_distance', 'last_mile_distance',
        'time_to_accept', 'time_to_pickup', 'session_time',
        'hour', 'is_peak_hour', 'lifetime_order_count',
        'cancel_rate', 'bike_issue_rate'
    ]
    
    # Test different labeling thresholds
    thresholds = {
        'Conservative': {'post_pickup_threshold': 0.8, 'bike_issue_threshold': 0.25},
        'Baseline': {'post_pickup_threshold': 0.7, 'bike_issue_threshold': 0.2},
        'Aggressive': {'post_pickup_threshold': 0.6, 'bike_issue_threshold': 0.15}
    }
    
    results = []
    
    for label_name, params in thresholds.items():
        print(f"\nTesting {label_name} labeling...")
        
        # Apply labeling criteria
        rider_metrics = calculate_rider_metrics_for_sensitivity(df)
        
        strategic_riders = rider_metrics[
            (rider_metrics['bike_issue_count'] >= 2) &
            (rider_metrics['post_pickup_rate'] > params['post_pickup_threshold']) &
            (rider_metrics['bike_issue_rate'] > params['bike_issue_threshold'])
        ].index
        
        # Create labels
        bike_issues['strategic'] = (
            bike_issues['rider_id'].isin(strategic_riders) & 
            (bike_issues['cancel_after_pickup'] == 1)
        ).astype(int)
        
        # Train model
        X = bike_issues[feature_cols].dropna()
        y = bike_issues.loc[X.index, 'strategic']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'label_type': label_name,
            'n_strategic': y.sum(),
            'strategic_rate': y.mean(),
            'auc': auc,
            'n_strategic_riders': len(strategic_riders)
        })
        
        print(f"  Strategic orders: {y.sum():,} ({y.mean():.1%})")
        print(f"  AUC: {auc:.3f}")
    
    return pd.DataFrame(results)

def calculate_rider_metrics_for_sensitivity(df):
    """Calculate rider metrics for sensitivity testing"""
    
    cancelled_orders = df[df['cancelled'] == 1]
    
    rider_metrics = pd.DataFrame()
    rider_metrics['total_orders'] = df.groupby('rider_id').size()
    rider_metrics['total_cancellations'] = cancelled_orders.groupby('rider_id').size()
    rider_metrics = rider_metrics.fillna({'total_cancellations': 0})
    
    bike_issues = cancelled_orders[cancelled_orders['reason_text'] == 'Cancel order due to bike issue']
    rider_metrics['bike_issue_count'] = bike_issues.groupby('rider_id').size()
    rider_metrics = rider_metrics.fillna({'bike_issue_count': 0})
    
    post_pickup = cancelled_orders[cancelled_orders['cancel_after_pickup'] == 1]
    rider_metrics['post_pickup_count'] = post_pickup.groupby('rider_id').size()
    rider_metrics = rider_metrics.fillna({'post_pickup_count': 0})
    
    rider_metrics['bike_issue_rate'] = rider_metrics['bike_issue_count'] / rider_metrics['total_orders']
    rider_metrics['post_pickup_rate'] = np.where(
        rider_metrics['total_cancellations'] > 0,
        rider_metrics['post_pickup_count'] / rider_metrics['total_cancellations'],
        0
    )
    
    return rider_metrics

def test_temporal_stability(df):
    """Test model stability over time"""
    print("\n" + "="*80)
    print("TEMPORAL STABILITY ANALYSIS")
    print("="*80)
    
    # Prepare data with temporal ordering
    df_sorted = df.sort_values('allot_time')
    bike_issues = df_sorted[df_sorted['reason_text'] == 'Cancel order due to bike issue'].copy()
    
    # Create strategic label
    bike_issues['strategic'] = (bike_issues['cancel_after_pickup'] == 1).astype(int)
    
    feature_cols = [
        'total_distance', 'first_mile_distance', 'last_mile_distance',
        'time_to_accept', 'time_to_pickup', 'session_time',
        'hour', 'is_peak_hour'
    ]
    
    # Clean data
    bike_issues_clean = bike_issues.dropna(subset=feature_cols + ['strategic'])
    X = bike_issues_clean[feature_cols]
    y = bike_issues_clean['strategic']
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        # Get time period
        train_start = bike_issues_clean.iloc[train_idx]['allot_time'].min()
        train_end = bike_issues_clean.iloc[train_idx]['allot_time'].max()
        test_start = bike_issues_clean.iloc[test_idx]['allot_time'].min()
        test_end = bike_issues_clean.iloc[test_idx]['allot_time'].max()
        
        results.append({
            'fold': fold + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'auc': auc,
            'test_strategic_rate': y_test.mean(),
            'train_period': f"{train_start[:10]} to {train_end[:10]}",
            'test_period': f"{test_start[:10]} to {test_end[:10]}"
        })
        
        print(f"\nFold {fold + 1}:")
        print(f"  Train period: {results[-1]['train_period']}")
        print(f"  Test period: {results[-1]['test_period']}")
        print(f"  AUC: {auc:.3f}")
    
    return pd.DataFrame(results)

def test_alternative_models(df):
    """Test alternative model specifications"""
    print("\n" + "="*80)
    print("ALTERNATIVE MODEL SPECIFICATIONS")
    print("="*80)
    
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue'].copy()
    bike_issues['strategic'] = (bike_issues['cancel_after_pickup'] == 1).astype(int)
    
    # Define different feature sets
    feature_sets = {
        'Minimal': ['total_distance', 'hour', 'is_peak_hour'],
        'Distance-focused': ['total_distance', 'first_mile_distance', 'last_mile_distance'],
        'Time-focused': ['time_to_accept', 'time_to_pickup', 'session_time', 'hour'],
        'Full': ['total_distance', 'first_mile_distance', 'last_mile_distance',
                'time_to_accept', 'time_to_pickup', 'session_time',
                'hour', 'is_peak_hour', 'lifetime_order_count']
    }
    
    results = []
    
    for name, features in feature_sets.items():
        print(f"\nTesting {name} feature set...")
        
        # Prepare data
        valid_features = [f for f in features if f in bike_issues.columns]
        X = bike_issues[valid_features].dropna()
        y = bike_issues.loc[X.index, 'strategic']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        # Feature importance
        top_features = sorted(zip(valid_features, model.feature_importances_), 
                            key=lambda x: x[1], reverse=True)[:3]
        
        results.append({
            'model': name,
            'n_features': len(valid_features),
            'auc': auc,
            'top_feature': top_features[0][0] if top_features else 'N/A',
            'top_importance': top_features[0][1] if top_features else 0
        })
        
        print(f"  Features: {len(valid_features)}")
        print(f"  AUC: {auc:.3f}")
        print(f"  Top feature: {results[-1]['top_feature']} ({results[-1]['top_importance']:.3f})")
    
    return pd.DataFrame(results)

def test_false_positive_analysis(df):
    """Analyze false positive patterns"""
    print("\n" + "="*80)
    print("FALSE POSITIVE ANALYSIS")
    print("="*80)
    
    # Get genuine bike issues (cancelled before pickup)
    genuine_bike_issues = df[
        (df['reason_text'] == 'Cancel order due to bike issue') & 
        (df['cancel_after_pickup'] == 0)
    ].copy()
    
    print(f"\nGenuine bike issues (pre-pickup): {len(genuine_bike_issues):,}")
    
    # Analyze characteristics
    print("\nGenuine bike issue characteristics:")
    print(f"  Average time to cancel: {genuine_bike_issues['time_to_cancel'].mean():.1f} min")
    print(f"  Average distance: {genuine_bike_issues['total_distance'].mean():.1f} km")
    print(f"  Peak hour rate: {genuine_bike_issues['is_peak_hour'].mean():.1%}")
    
    # Compare with strategic (post-pickup)
    strategic_bike_issues = df[
        (df['reason_text'] == 'Cancel order due to bike issue') & 
        (df['cancel_after_pickup'] == 1)
    ]
    
    # Statistical tests
    from scipy import stats
    
    # T-test for time to cancel
    if 'time_to_cancel' in genuine_bike_issues.columns:
        genuine_times = genuine_bike_issues['time_to_cancel'].dropna()
        strategic_times = strategic_bike_issues['time_to_cancel'].dropna()
        
        if len(genuine_times) > 0 and len(strategic_times) > 0:
            t_stat, p_value = stats.ttest_ind(genuine_times, strategic_times)
            print(f"\nTime to cancel comparison:")
            print(f"  Genuine mean: {genuine_times.mean():.1f} min")
            print(f"  Strategic mean: {strategic_times.mean():.1f} min")
            print(f"  T-test p-value: {p_value:.4f}")
    
    return genuine_bike_issues

def create_robustness_plots(label_results, temporal_results, model_results):
    """Create robustness check visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Label sensitivity
    ax1 = axes[0, 0]
    
    x = np.arange(len(label_results))
    width = 0.35
    
    ax1.bar(x - width/2, label_results['n_strategic'], width, 
            label='Strategic Orders', alpha=0.7)
    ax1.bar(x + width/2, label_results['n_strategic_riders'], width,
            label='Strategic Riders', alpha=0.7)
    
    ax1.set_xlabel('Labeling Threshold', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Label Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(label_results['label_type'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add AUC values
    for i, (idx, row) in enumerate(label_results.iterrows()):
        ax1.text(i, row['n_strategic'] + 50, f"AUC: {row['auc']:.3f}", 
                ha='center', fontsize=9)
    
    # Plot 2: Temporal stability
    ax2 = axes[0, 1]
    
    ax2.plot(temporal_results['fold'], temporal_results['auc'], 
            'o-', linewidth=2, markersize=10, color='darkblue')
    ax2.axhline(y=temporal_results['auc'].mean(), color='red', 
               linestyle='--', alpha=0.7, 
               label=f"Mean AUC: {temporal_results['auc'].mean():.3f}")
    
    ax2.set_xlabel('Time Fold', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Temporal Stability of Model', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.6, 0.8)
    
    # Plot 3: Model comparison
    ax3 = axes[1, 0]
    
    ax3.barh(model_results['model'], model_results['auc'], 
            color=['red', 'orange', 'yellow', 'green'])
    
    # Add value labels
    for idx, row in model_results.iterrows():
        ax3.text(row['auc'] + 0.005, idx, f"{row['auc']:.3f}", 
                va='center', fontsize=10)
    
    ax3.set_xlabel('AUC', fontsize=12)
    ax3.set_title('Alternative Model Specifications', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0.5, 0.8)
    
    # Plot 4: Feature count vs performance
    ax4 = axes[1, 1]
    
    ax4.scatter(model_results['n_features'], model_results['auc'], 
               s=200, alpha=0.7, c=range(len(model_results)), cmap='viridis')
    
    # Add labels
    for idx, row in model_results.iterrows():
        ax4.annotate(row['model'], 
                    (row['n_features'], row['auc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Number of Features', fontsize=12)
    ax4.set_ylabel('AUC', fontsize=12)
    ax4.set_title('Model Complexity vs Performance', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/robustness_checks.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Generated figures/robustness_checks.png")

def generate_robustness_tables(label_results, temporal_results, model_results):
    """Generate LaTeX tables for robustness results"""
    
    # Table 1: Label sensitivity
    label_table = """
\\begin{table}[H]
\\centering
\\caption{Label Sensitivity Analysis Results}
\\label{tab:label_sensitivity}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Threshold} & \\textbf{Strategic Orders} & \\textbf{Strategic Rate} & \\textbf{AUC} & \\textbf{Strategic Riders} \\\\
\\midrule
"""
    
    for _, row in label_results.iterrows():
        label_table += f"{row['label_type']} & {row['n_strategic']:,} & "
        label_table += f"{row['strategic_rate']:.1%} & {row['auc']:.3f} & "
        label_table += f"{row['n_strategic_riders']:,} \\\\\n"
    
    label_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_label_sensitivity.tex', 'w') as f:
        f.write(label_table)
    
    print("✓ Generated tables/table_label_sensitivity.tex")
    
    # Table 2: Temporal stability
    temporal_table = """
\\begin{table}[H]
\\centering
\\caption{Temporal Stability Analysis}
\\label{tab:temporal_stability}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Fold} & \\textbf{Train Size} & \\textbf{Test Size} & \\textbf{AUC} \\\\
\\midrule
"""
    
    for _, row in temporal_results.iterrows():
        temporal_table += f"{row['fold']} & {row['train_size']:,} & "
        temporal_table += f"{row['test_size']:,} & {row['auc']:.3f} \\\\\n"
    
    temporal_table += f"""\\midrule
Mean & -- & -- & {temporal_results['auc'].mean():.3f} \\\\
Std Dev & -- & -- & {temporal_results['auc'].std():.3f} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_temporal_stability.tex', 'w') as f:
        f.write(temporal_table)
    
    print("✓ Generated tables/table_temporal_stability.tex")
    
    # Table 3: Model specifications
    model_table = """
\\begin{table}[H]
\\centering
\\caption{Alternative Model Specifications}
\\label{tab:model_specifications}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Model} & \\textbf{Features} & \\textbf{AUC} & \\textbf{Top Feature} \\\\
\\midrule
"""
    
    for _, row in model_results.iterrows():
        model_table += f"{row['model']} & {row['n_features']} & "
        model_table += f"{row['auc']:.3f} & {row['top_feature'].replace('_', ' ')} \\\\\n"
    
    model_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_model_specifications.tex', 'w') as f:
        f.write(model_table)
    
    print("✓ Generated tables/table_model_specifications.tex")

def main():
    """Main execution function"""
    
    # Create output directories
    import os
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("="*80)
    print("ROBUSTNESS CHECKS AND SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('shadowfax_processed-data-final.csv')
    
    # Run robustness tests
    label_results = test_label_sensitivity(df)
    temporal_results = test_temporal_stability(df)
    model_results = test_alternative_models(df)
    genuine_analysis = test_false_positive_analysis(df)
    
    # Generate outputs
    create_robustness_plots(label_results, temporal_results, model_results)
    generate_robustness_tables(label_results, temporal_results, model_results)
    
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print(f"  - Model AUC stable across label thresholds: {label_results['auc'].std():.3f} std dev")
    print(f"  - Temporal stability: {temporal_results['auc'].mean():.3f} ± {temporal_results['auc'].std():.3f}")
    print(f"  - Best feature set: {model_results.loc[model_results['auc'].idxmax(), 'model']}")
    print("\nGenerated files:")
    print("  Tables:")
    print("    - tables/table_label_sensitivity.tex")
    print("    - tables/table_temporal_stability.tex") 
    print("    - tables/table_model_specifications.tex")
    print("  Figures:")
    print("    - figures/robustness_checks.png")

if __name__ == "__main__":
    main()