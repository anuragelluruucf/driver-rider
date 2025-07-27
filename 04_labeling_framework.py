#!/usr/bin/env python3
"""
04_labeling_framework.py
Strategic Detection Framework with behavioral proxies
Generates Venn diagram and logic documentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import seaborn as sns

def apply_strategic_labeling(df):
    """
    Apply the three-criteria strategic labeling framework
    Returns labeled dataframe and statistics
    """
    print("="*80)
    print("STRATEGIC DETECTION FRAMEWORK")
    print("="*80)
    
    # Calculate rider-level metrics
    rider_metrics = calculate_rider_metrics(df)
    
    # Apply three criteria
    criterion1 = rider_metrics['bike_issue_count'] >= 2
    criterion2 = rider_metrics['post_pickup_rate'] > 0.7
    criterion3 = rider_metrics['bike_issue_rate'] > 0.2
    
    # Strategic riders meet ALL three criteria
    strategic_riders = rider_metrics[criterion1 & criterion2 & criterion3].index
    
    # Calculate set sizes for Venn diagram
    only_c1 = len(rider_metrics[criterion1 & ~criterion2 & ~criterion3])
    only_c2 = len(rider_metrics[~criterion1 & criterion2 & ~criterion3])
    only_c3 = len(rider_metrics[~criterion1 & ~criterion2 & criterion3])
    c1_c2 = len(rider_metrics[criterion1 & criterion2 & ~criterion3])
    c1_c3 = len(rider_metrics[criterion1 & ~criterion2 & criterion3])
    c2_c3 = len(rider_metrics[~criterion1 & criterion2 & criterion3])
    all_three = len(strategic_riders)
    
    print(f"\nCriterion 1 (≥2 bike issues): {criterion1.sum():,} riders")
    print(f"Criterion 2 (>70% post-pickup): {criterion2.sum():,} riders")
    print(f"Criterion 3 (>20% bike issue rate): {criterion3.sum():,} riders")
    print(f"\nStrategic riders (all 3 criteria): {all_three:,} riders")
    
    # Label orders
    df['strategic_rider'] = df['rider_id'].isin(strategic_riders).astype(int)
    df['strategic_order'] = (
        df['strategic_rider'] & 
        (df['reason_text'] == 'Cancel order due to bike issue') &
        (df['cancel_after_pickup'] == 1)
    ).astype(int)
    
    print(f"\nStrategic orders labeled: {df['strategic_order'].sum():,}")
    
    return df, (only_c1, only_c2, only_c3, c1_c2, c1_c3, c2_c3, all_three)

def calculate_rider_metrics(df):
    """Calculate rider-level behavioral metrics"""
    
    # Get all cancelled orders
    cancelled_orders = df[df['cancelled'] == 1]
    
    # Calculate metrics
    rider_metrics = pd.DataFrame()
    
    # Total orders per rider
    rider_metrics['total_orders'] = df.groupby('rider_id').size()
    
    # Total cancellations
    rider_metrics['total_cancellations'] = cancelled_orders.groupby('rider_id').size()
    rider_metrics = rider_metrics.fillna({'total_cancellations': 0})
    
    # Bike issue cancellations
    bike_issues = cancelled_orders[cancelled_orders['reason_text'] == 'Cancel order due to bike issue']
    rider_metrics['bike_issue_count'] = bike_issues.groupby('rider_id').size()
    rider_metrics = rider_metrics.fillna({'bike_issue_count': 0})
    
    # Post-pickup cancellations
    post_pickup = cancelled_orders[cancelled_orders['cancel_after_pickup'] == 1]
    rider_metrics['post_pickup_count'] = post_pickup.groupby('rider_id').size()
    rider_metrics = rider_metrics.fillna({'post_pickup_count': 0})
    
    # Calculate rates
    rider_metrics['cancel_rate'] = rider_metrics['total_cancellations'] / rider_metrics['total_orders']
    rider_metrics['bike_issue_rate'] = rider_metrics['bike_issue_count'] / rider_metrics['total_orders']
    
    # Post-pickup rate (among cancellations)
    rider_metrics['post_pickup_rate'] = np.where(
        rider_metrics['total_cancellations'] > 0,
        rider_metrics['post_pickup_count'] / rider_metrics['total_cancellations'],
        0
    )
    
    return rider_metrics

def create_venn_diagram(set_sizes):
    """Create Venn diagram for strategic classification"""
    
    plt.figure(figsize=(10, 8))
    
    # Unpack set sizes
    only_c1, only_c2, only_c3, c1_c2, c1_c3, c2_c3, all_three = set_sizes
    
    # Create Venn diagram
    v = venn3(subsets=(only_c1, only_c2, c1_c2, only_c3, c1_c3, c2_c3, all_three), 
              set_labels=('≥2 Bike Issues', '>70% Post-Pickup', '>20% Bike Issue Rate'))
    
    # Customize colors
    v.get_patch_by_id('100').set_color('lightblue')
    v.get_patch_by_id('010').set_color('lightcoral') 
    v.get_patch_by_id('001').set_color('lightgreen')
    v.get_patch_by_id('110').set_color('lightyellow')
    v.get_patch_by_id('101').set_color('lightcyan')
    v.get_patch_by_id('011').set_color('lightpink')
    v.get_patch_by_id('111').set_color('red')
    v.get_patch_by_id('111').set_alpha(0.8)
    
    # Add strategic riders label
    plt.text(0, -0.05, f'Strategic\\nRiders\\n({all_three})', 
             fontsize=14, fontweight='bold', ha='center', va='center', color='white',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="darkred", alpha=0.8))
    
    # Add circles
    c = venn3_circles(subsets=(only_c1, only_c2, c1_c2, only_c3, c1_c3, c2_c3, all_three), 
                      linewidth=2)
    
    plt.title('Strategic Rider Classification Framework', fontsize=16, fontweight='bold', pad=20)
    plt.text(0, -0.8, 'All three criteria must be met for strategic classification', 
             fontsize=12, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('figures/strategic_classification_venn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Generated figures/strategic_classification_venn.png")

def generate_labeling_logic_table():
    """Generate LaTeX table explaining labeling logic"""
    
    logic_table = """
\\begin{table}[H]
\\centering
\\caption{Strategic Behavior Detection Criteria}
\\label{tab:labeling_logic}
\\begin{tabular}{p{3cm}p{7cm}p{4cm}}
\\toprule
\\textbf{Criterion} & \\textbf{Definition} & \\textbf{Behavioral Proxy} \\\\
\\midrule
\\textbf{Criterion 1} & Rider has $\\geq$ 2 bike issue cancellations & Repeated behavior indicates pattern \\\\
\\\\
\\textbf{Criterion 2} & Rider cancels >70\\% of orders after pickup & Suggests exploitation of verification gap \\\\
\\\\
\\textbf{Criterion 3} & Bike issues constitute >20\\% of rider's total orders & Disproportionate use of unverifiable excuse \\\\
\\midrule
\\textbf{Strategic Label} & \\multicolumn{2}{l}{Applied when ALL THREE criteria are satisfied} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_labeling_logic.tex', 'w') as f:
        f.write(logic_table)
    
    print("✓ Generated tables/table_labeling_logic.tex")

def analyze_labeled_data(df):
    """Analyze the labeled dataset"""
    
    print("\n" + "="*50)
    print("LABELED DATA ANALYSIS")
    print("="*50)
    
    # Overall statistics
    total_riders = df['rider_id'].nunique()
    strategic_riders = df[df['strategic_rider'] == 1]['rider_id'].nunique()
    strategic_orders = df['strategic_order'].sum()
    
    print(f"\nTotal riders: {total_riders:,}")
    print(f"Strategic riders: {strategic_riders:,} ({strategic_riders/total_riders*100:.1f}%)")
    print(f"Strategic orders: {strategic_orders:,}")
    
    # Analyze strategic vs non-strategic patterns
    strategic_df = df[df['strategic_order'] == 1]
    
    if len(strategic_df) > 0:
        print(f"\nStrategic order characteristics:")
        print(f"  Average distance: {strategic_df['total_distance'].mean():.1f} km")
        print(f"  Peak hour rate: {strategic_df['is_peak_hour'].mean():.1%}")
        print(f"  Average time to cancel: {strategic_df['time_to_cancel'].mean():.1f} min")
    
    # Generate comparison table
    comparison_table = """
\\begin{table}[H]
\\centering
\\caption{Strategic vs Non-Strategic Order Characteristics}
\\label{tab:strategic_comparison}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Characteristic} & \\textbf{Strategic} & \\textbf{Non-Strategic} \\\\
\\midrule
"""
    
    # Calculate comparisons for bike issues only
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue']
    strategic_bike = bike_issues[bike_issues['strategic_order'] == 1]
    non_strategic_bike = bike_issues[bike_issues['strategic_order'] == 0]
    
    if len(strategic_bike) > 0 and len(non_strategic_bike) > 0:
        comparisons = [
            ('Average distance (km)', 
             f"{strategic_bike['total_distance'].mean():.1f}",
             f"{non_strategic_bike['total_distance'].mean():.1f}"),
            ('Peak hour rate', 
             f"{strategic_bike['is_peak_hour'].mean():.1%}",
             f"{non_strategic_bike['is_peak_hour'].mean():.1%}"),
            ('Post-pickup rate', 
             f"{strategic_bike['cancel_after_pickup'].mean():.1%}",
             f"{non_strategic_bike['cancel_after_pickup'].mean():.1%}"),
            ('Avg time to cancel (min)', 
             f"{strategic_bike['time_to_cancel'].mean():.1f}",
             f"{non_strategic_bike['time_to_cancel'].mean():.1f}"),
            ('Sample size', 
             f"{len(strategic_bike):,}",
             f"{len(non_strategic_bike):,}")
        ]
        
        for metric, strat_val, non_strat_val in comparisons:
            comparison_table += f"{metric} & {strat_val} & {non_strat_val} \\\\\n"
    
    comparison_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_strategic_comparison.tex', 'w') as f:
        f.write(comparison_table)
    
    print("\n✓ Generated tables/table_strategic_comparison.tex")

def create_validation_plots(df):
    """Create plots to validate labeling framework"""
    
    print("\n" + "="*50)
    print("CREATING VALIDATION PLOTS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of bike issue counts
    ax1 = axes[0, 0]
    rider_bike_counts = df[df['reason_text'] == 'Cancel order due to bike issue'].groupby('rider_id').size()
    rider_bike_counts.value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Bike Issues per Rider')
    ax1.set_ylabel('Number of Riders')
    ax1.set_title('Distribution of Bike Issue Counts')
    ax1.text(2, ax1.get_ylim()[1]*0.8, 'Threshold ≥2', color='red', fontsize=10)
    
    # Plot 2: Post-pickup rates distribution
    ax2 = axes[0, 1]
    rider_metrics = calculate_rider_metrics(df)
    ax2.hist(rider_metrics['post_pickup_rate'], bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Post-Pickup Cancellation Rate')
    ax2.set_ylabel('Number of Riders')
    ax2.set_title('Distribution of Post-Pickup Rates')
    ax2.text(0.72, ax2.get_ylim()[1]*0.8, 'Threshold >70%', color='red', fontsize=10)
    
    # Plot 3: Bike issue rates distribution
    ax3 = axes[1, 0]
    ax3.hist(rider_metrics['bike_issue_rate'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(x=0.2, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Bike Issue Rate (% of Total Orders)')
    ax3.set_ylabel('Number of Riders')
    ax3.set_title('Distribution of Bike Issue Rates')
    ax3.text(0.22, ax3.get_ylim()[1]*0.8, 'Threshold >20%', color='red', fontsize=10)
    
    # Plot 4: Strategic classification results
    ax4 = axes[1, 1]
    classification_counts = pd.Series({
        'Strategic': df[df['strategic_rider'] == 1]['rider_id'].nunique(),
        'Non-Strategic': df[df['strategic_rider'] == 0]['rider_id'].nunique()
    })
    classification_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', 
                              colors=['red', 'lightgray'], startangle=90)
    ax4.set_ylabel('')
    ax4.set_title('Final Classification Results')
    
    plt.tight_layout()
    plt.savefig('figures/labeling_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated figures/labeling_validation.png")

def export_labeled_dataset(df):
    """Export the labeled dataset"""
    
    # Save full labeled dataset
    df.to_csv('data/labeled_orders.csv', index=False)
    print("\n✓ Exported data/labeled_orders.csv")
    
    # Save strategic orders only
    strategic_orders = df[df['strategic_order'] == 1]
    strategic_orders.to_csv('data/strategic_orders.csv', index=False)
    print(f"✓ Exported data/strategic_orders.csv ({len(strategic_orders):,} orders)")
    
    # Save rider metrics
    rider_metrics = calculate_rider_metrics(df)
    rider_metrics.to_csv('data/rider_metrics.csv')
    print(f"✓ Exported data/rider_metrics.csv ({len(rider_metrics):,} riders)")

def main():
    """Main execution function"""
    
    # Create output directories
    import os
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("="*80)
    print("STRATEGIC DETECTION FRAMEWORK")
    print("="*80)
    
    # Load data
    df = pd.read_csv('shadowfax_processed-data-final.csv')
    print(f"\nLoaded {len(df):,} orders")
    
    # Apply labeling framework
    df, set_sizes = apply_strategic_labeling(df)
    
    # Create visualizations
    create_venn_diagram(set_sizes)
    create_validation_plots(df)
    
    # Generate documentation
    generate_labeling_logic_table()
    analyze_labeled_data(df)
    
    # Export results
    export_labeled_dataset(df)
    
    print("\n" + "="*80)
    print("LABELING FRAMEWORK COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  Tables:")
    print("    - tables/table_labeling_logic.tex")
    print("    - tables/table_strategic_comparison.tex")
    print("  Figures:")
    print("    - figures/strategic_classification_venn.png")
    print("    - figures/labeling_validation.png")
    print("  Data:")
    print("    - data/labeled_orders.csv")
    print("    - data/strategic_orders.csv")
    print("    - data/rider_metrics.csv")

if __name__ == "__main__":
    main()