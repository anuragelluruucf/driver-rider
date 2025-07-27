#!/usr/bin/env python3
"""
01_data_exploration.py
Initial data exploration and summary statistics
Generates LaTeX tables for the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

def load_and_explore_data():
    """Load data and perform initial exploration"""
    print("="*80)
    print("DATA EXPLORATION AND SUMMARY STATISTICS")
    print("="*80)
    
    # Load main dataset
    df = pd.read_csv('shadowfax_processed-data-final.csv')
    print(f"\nDataset shape: {df.shape}")
    print(f"Total orders: {len(df):,}")
    print(f"Total riders: {df['rider_id'].nunique():,}")
    
    # Basic statistics
    print("\n" + "="*50)
    print("CANCELLATION STATISTICS")
    print("="*50)
    
    cancel_rate = df['cancelled'].mean()
    print(f"Overall cancellation rate: {cancel_rate:.2%}")
    
    # Cancellation reasons
    cancelled_df = df[df['cancelled'] == 1]
    print(f"\nTotal cancellations: {len(cancelled_df):,}")
    
    # Reason distribution
    print("\nCancellation Reason Distribution:")
    reason_counts = cancelled_df['reason_text'].value_counts()
    for reason, count in reason_counts.items():
        pct = count / len(cancelled_df) * 100
        print(f"  {reason}: {count:,} ({pct:.1f}%)")
    
    # Bike issues analysis
    bike_issues = cancelled_df[cancelled_df['reason_text'] == 'Cancel order due to bike issue']
    print(f"\nBike issue cancellations: {len(bike_issues):,}")
    print(f"Bike issues as % of all cancellations: {len(bike_issues)/len(cancelled_df)*100:.1f}%")
    
    # Post-pickup analysis
    bike_post_pickup = bike_issues['cancel_after_pickup'].mean()
    all_post_pickup = cancelled_df['cancel_after_pickup'].mean()
    print(f"\nPost-pickup rates:")
    print(f"  Bike issues: {bike_post_pickup:.1%}")
    print(f"  All cancellations: {all_post_pickup:.1%}")
    
    return df

def generate_variable_summary_table(df):
    """Generate LaTeX table of variable definitions"""
    
    variables = {
        'order_id': 'Unique identifier for each order',
        'rider_id': 'Unique identifier for each delivery rider',
        'cancelled': 'Binary indicator (1 if order cancelled, 0 otherwise)',
        'reason_text': 'Text description of cancellation reason',
        'allot_time': 'Time when order was allocated to rider',
        'accept_time': 'Time when rider accepted the order',
        'pickup_time': 'Time when rider picked up the order',
        'cancelled_time': 'Time when order was cancelled',
        'customer_lat': 'Customer location latitude',
        'customer_lon': 'Customer location longitude',
        'rider_lat': 'Rider location latitude at allocation',
        'rider_lon': 'Rider location longitude at allocation',
        'first_mile_distance': 'Distance from rider to restaurant (km)',
        'last_mile_distance': 'Distance from restaurant to customer (km)',
        'total_distance': 'Total delivery distance (km)',
        'cancel_after_pickup': 'Binary indicator for post-pickup cancellation'
    }
    
    # Create LaTeX table
    latex_table = """
\\begin{table}[H]
\\centering
\\caption{Dataset Variable Definitions}
\\label{tab:variables}
\\begin{tabular}{p{4cm}p{10cm}}
\\toprule
\\textbf{Variable Name} & \\textbf{Description} \\\\
\\midrule
"""
    
    for var, desc in variables.items():
        latex_table += f"{var.replace('_', ' ')} & {desc} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save to file
    with open('tables/table_variables.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nGenerated tables/table_variables.tex")
    
    return latex_table

def generate_summary_statistics_table(df):
    """Generate LaTeX table of summary statistics"""
    
    # Select key numerical variables
    numerical_vars = ['total_distance', 'first_mile_distance', 'last_mile_distance',
                     'time_to_accept', 'time_to_pickup', 'session_time']
    
    # Calculate statistics
    stats_data = []
    for var in numerical_vars:
        if var in df.columns:
            stats_data.append({
                'Variable': var.replace('_', ' ').title(),
                'Mean': df[var].mean(),
                'Std Dev': df[var].std(),
                'Min': df[var].min(),
                'Max': df[var].max(),
                'N': df[var].notna().sum()
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create LaTeX table
    latex_table = """
\\begin{table}[H]
\\centering
\\caption{Summary Statistics for Key Variables}
\\label{tab:summary_stats}
\\begin{tabular}{lr@{.}lr@{.}lr@{.}lr@{.}lr}
\\toprule
\\textbf{Variable} & \\multicolumn{2}{c}{\\textbf{Mean}} & \\multicolumn{2}{c}{\\textbf{Std Dev}} & \\multicolumn{2}{c}{\\textbf{Min}} & \\multicolumn{2}{c}{\\textbf{Max}} & \\textbf{N} \\\\
\\midrule
"""
    
    for _, row in stats_df.iterrows():
        # Split numbers at decimal point
        mean_int, mean_dec = f"{row['Mean']:.2f}".split('.')
        std_int, std_dec = f"{row['Std Dev']:.2f}".split('.')
        min_int, min_dec = f"{row['Min']:.2f}".split('.')
        max_int, max_dec = f"{row['Max']:.2f}".split('.')
        
        latex_table += f"{row['Variable']} & {mean_int}&{mean_dec} & {std_int}&{std_dec} & "
        latex_table += f"{min_int}&{min_dec} & {max_int}&{max_dec} & {row['N']:,} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save to file
    with open('tables/table_summary_stats.tex', 'w') as f:
        f.write(latex_table)
    
    print("Generated tables/table_summary_stats.tex")
    
    return latex_table

def analyze_missing_data(df):
    """Analyze and report missing data"""
    
    print("\n" + "="*50)
    print("MISSING DATA ANALYSIS")
    print("="*50)
    
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Variable': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing %': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing_df) > 0:
        print("\nVariables with missing data:")
        for _, row in missing_df.iterrows():
            print(f"  {row['Variable']}: {row['Missing Count']:,} ({row['Missing %']:.1f}%)")
    else:
        print("\nNo missing data found in the dataset")
    
    # Generate LaTeX table for missing data
    if len(missing_df) > 0:
        latex_table = """
\\begin{table}[H]
\\centering
\\caption{Missing Data Summary}
\\label{tab:missing_data}
\\begin{tabular}{lrr}
\\toprule
\\textbf{Variable} & \\textbf{Missing Count} & \\textbf{Missing \\%} \\\\
\\midrule
"""
        
        for _, row in missing_df.iterrows():
            latex_table += f"{row['Variable'].replace('_', ' ')} & {row['Missing Count']:,} & {row['Missing %']:.1f}\\% \\\\\n"
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open('tables/table_missing_data.tex', 'w') as f:
            f.write(latex_table)
        
        print("\nGenerated tables/table_missing_data.tex")

def create_exploration_plots(df):
    """Create exploration plots"""
    
    print("\n" + "="*50)
    print("GENERATING EXPLORATION PLOTS")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Cancellation rate by hour
    plt.figure(figsize=(12, 6))
    hourly_cancel = df.groupby('hour')['cancelled'].agg(['mean', 'count'])
    
    plt.subplot(1, 2, 1)
    plt.bar(hourly_cancel.index, hourly_cancel['mean'], alpha=0.7, color='coral')
    plt.xlabel('Hour of Day')
    plt.ylabel('Cancellation Rate')
    plt.title('Cancellation Rate by Hour')
    plt.xticks(range(0, 24))
    
    # 2. Distance distribution
    plt.subplot(1, 2, 2)
    df['total_distance'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Total Distance (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delivery Distances')
    plt.xlim(0, 20)
    
    plt.tight_layout()
    plt.savefig('figures/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated figures/data_exploration.png")

def export_sample_data(df):
    """Export sample data for documentation"""
    
    # Create a sample of the data
    sample_df = df.head(100)
    
    # Save as CSV
    sample_df.to_csv('data/sample_data.csv', index=False)
    print("\nExported data/sample_data.csv")
    
    # Create summary statistics CSV
    summary_stats = df.describe()
    summary_stats.to_csv('data/summary_statistics.csv')
    print("Exported data/summary_statistics.csv")

def main():
    """Main execution function"""
    
    # Create output directories
    import os
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Generate tables
    generate_variable_summary_table(df)
    generate_summary_statistics_table(df)
    analyze_missing_data(df)
    
    # Create plots
    create_exploration_plots(df)
    
    # Export sample data
    export_sample_data(df)
    
    print("\n" + "="*80)
    print("DATA EXPLORATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - tables/table_variables.tex")
    print("  - tables/table_summary_stats.tex")
    print("  - tables/table_missing_data.tex (if applicable)")
    print("  - figures/data_exploration.png")
    print("  - data/sample_data.csv")
    print("  - data/summary_statistics.csv")

if __name__ == "__main__":
    main()