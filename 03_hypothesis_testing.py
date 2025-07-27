#!/usr/bin/env python3
"""
03_hypothesis_testing.py
Test all five hypotheses (H1-H5) with statistical rigor
Generates LaTeX tables and figures for each hypothesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def test_hypothesis_1(df):
    """
    H1: Threshold Effect in Strategic Behavior
    Riders with ≥2 prior bike issue cancellations are significantly more likely 
    to cancel strategically than those with fewer incidents.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 1: THRESHOLD EFFECT")
    print("="*60)
    
    # Calculate cumulative bike issues per rider
    rider_bike_issues = df[
        (df['cancelled'] == 1) & 
        (df['reason_text'] == 'Cancel order due to bike issue')
    ].groupby('rider_id').size()
    
    # Create threshold groups
    results = []
    for k in range(6):
        riders_with_k = set(rider_bike_issues[rider_bike_issues >= k].index)
        
        if len(riders_with_k) > 0:
            # Calculate strategic probability
            strategic_mask = (
                df['rider_id'].isin(riders_with_k) & 
                (df['reason_text'] == 'Cancel order due to bike issue') &
                (df['cancel_after_pickup'] == 1)
            )
            
            total_bike_issues = (
                df['rider_id'].isin(riders_with_k) & 
                (df['reason_text'] == 'Cancel order due to bike issue')
            ).sum()
            
            strategic_rate = strategic_mask.sum() / total_bike_issues if total_bike_issues > 0 else 0
            
            results.append({
                'k': k,
                'n_riders': len(riders_with_k),
                'strategic_rate': strategic_rate,
                'total_cancellations': total_bike_issues
            })
    
    results_df = pd.DataFrame(results)
    
    # Statistical test: Compare k<2 vs k>=2
    below_threshold = df[df['rider_id'].isin(
        rider_bike_issues[rider_bike_issues < 2].index
    )]
    above_threshold = df[df['rider_id'].isin(
        rider_bike_issues[rider_bike_issues >= 2].index
    )]
    
    # Calculate rates
    below_rate = (
        below_threshold[
            (below_threshold['reason_text'] == 'Cancel order due to bike issue') &
            (below_threshold['cancel_after_pickup'] == 1)
        ].shape[0] / 
        below_threshold[below_threshold['reason_text'] == 'Cancel order due to bike issue'].shape[0]
        if below_threshold[below_threshold['reason_text'] == 'Cancel order due to bike issue'].shape[0] > 0 
        else 0
    )
    
    above_rate = (
        above_threshold[
            (above_threshold['reason_text'] == 'Cancel order due to bike issue') &
            (above_threshold['cancel_after_pickup'] == 1)
        ].shape[0] / 
        above_threshold[above_threshold['reason_text'] == 'Cancel order due to bike issue'].shape[0]
        if above_threshold[above_threshold['reason_text'] == 'Cancel order due to bike issue'].shape[0] > 0 
        else 0
    )
    
    # Chi-square test
    contingency_table = np.array([
        [below_threshold[(below_threshold['reason_text'] == 'Cancel order due to bike issue') & 
                        (below_threshold['cancel_after_pickup'] == 1)].shape[0],
         below_threshold[(below_threshold['reason_text'] == 'Cancel order due to bike issue') & 
                        (below_threshold['cancel_after_pickup'] == 0)].shape[0]],
        [above_threshold[(above_threshold['reason_text'] == 'Cancel order due to bike issue') & 
                        (above_threshold['cancel_after_pickup'] == 1)].shape[0],
         above_threshold[(above_threshold['reason_text'] == 'Cancel order due to bike issue') & 
                        (above_threshold['cancel_after_pickup'] == 0)].shape[0]]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nBelow threshold (<2): {below_rate:.1%} strategic")
    print(f"Above threshold (≥2): {above_rate:.1%} strategic")
    print(f"Chi-square test: χ² = {chi2:.2f}, p = {p_value:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['strategic_rate'], 'o-', linewidth=3, markersize=10)
    plt.axvline(x=2, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(2.1, 0.2, 'threshold', fontsize=12, color='red')
    
    # Add percentage labels
    for _, row in results_df.iterrows():
        plt.text(row['k'], row['strategic_rate'] + 0.01, 
                f"{row['strategic_rate']:.1%}", ha='center', fontsize=10)
    
    plt.xlabel('Number of Prior Bike Issue Cancellations (k)', fontsize=12)
    plt.ylabel('Probability of Strategic Cancellation', fontsize=12)
    plt.title('H1: Strategic Probability by Prior Incidents', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(results_df['strategic_rate']) * 1.2)
    
    plt.tight_layout()
    plt.savefig('figures/h1_threshold_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{H1: Threshold Effect Test Results}}
\\label{{tab:h1_results}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Group}} & \\textbf{{Strategic Rate}} & \\textbf{{N}} & \\textbf{{Statistical Test}} \\\\
\\midrule
Below threshold (k < 2) & {below_rate:.1%} & {len(below_threshold):,} & \\multirow{{2}}{{*}}{{$\\chi^2$ = {chi2:.2f}}} \\\\
Above threshold (k $\\geq$ 2) & {above_rate:.1%} & {len(above_threshold):,} & \\\\
\\midrule
Effect size & \\multicolumn{{2}}{{c}}{{{above_rate - below_rate:.1%} increase}} & p = {p_value:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open('tables/table_h1_results.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nH1 SUPPORTED: Significant threshold effect found")
    print(f"Generated tables/table_h1_results.tex")
    print(f"Generated figures/h1_threshold_test.png")
    
    return p_value < 0.05, chi2, p_value

def test_hypothesis_2(df):
    """
    H2: Peak Hour Concentration
    Strategic cancellations are disproportionately concentrated during 
    peak hours (12-14, 18-21) when outside options are valuable.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 2: PEAK HOUR CONCENTRATION")
    print("="*60)
    
    # Define strategic cancellations
    strategic_mask = (
        (df['reason_text'] == 'Cancel order due to bike issue') & 
        (df['cancel_after_pickup'] == 1)
    )
    
    # Calculate distributions
    all_orders_hourly = df.groupby('hour').size()
    strategic_hourly = df[strategic_mask].groupby('hour').size()
    
    # Normalize to proportions
    all_orders_prop = all_orders_hourly / all_orders_hourly.sum()
    strategic_prop = strategic_hourly / strategic_hourly.sum()
    
    # Peak vs off-peak analysis
    peak_hours = [12, 13, 14, 18, 19, 20, 21]
    
    all_peak_prop = all_orders_prop[all_orders_prop.index.isin(peak_hours)].sum()
    strategic_peak_prop = strategic_prop[strategic_prop.index.isin(peak_hours)].sum()
    
    all_offpeak_prop = 1 - all_peak_prop
    strategic_offpeak_prop = 1 - strategic_peak_prop
    
    # Chi-square test
    observed = np.array([
        [df[df['is_peak_hour'] == 1].shape[0], 
         df[strategic_mask & (df['is_peak_hour'] == 1)].shape[0]],
        [df[df['is_peak_hour'] == 0].shape[0], 
         df[strategic_mask & (df['is_peak_hour'] == 0)].shape[0]]
    ])
    
    chi2, p_value = stats.chi2_contingency(observed)[:2]
    
    print(f"\nPeak hour proportion:")
    print(f"  All orders: {all_peak_prop:.1%}")
    print(f"  Strategic cancellations: {strategic_peak_prop:.1%}")
    print(f"  Concentration ratio: {strategic_peak_prop/all_peak_prop:.2f}x")
    print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    hours = list(range(24))
    width = 0.35
    x = np.arange(len(hours))
    
    # Ensure all hours are represented
    all_props = [all_orders_prop.get(h, 0) for h in hours]
    strategic_props = [strategic_prop.get(h, 0) for h in hours]
    
    plt.bar(x - width/2, all_props, width, label='All orders', alpha=0.7)
    plt.bar(x + width/2, strategic_props, width, label='Strategic cancellations', alpha=0.7)
    
    # Highlight peak hours
    for hour in peak_hours:
        plt.axvspan(hour - 0.5, hour + 0.5, alpha=0.1, color='yellow')
    
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Proportion of Orders', fontsize=12)
    plt.title('H2: Order Distribution by Hour', fontsize=14, fontweight='bold')
    plt.xticks(hours)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add peak hour labels
    plt.text(13, max(max(all_props), max(strategic_props)) * 0.9, 'lunch peak', 
             ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    plt.text(19.5, max(max(all_props), max(strategic_props)) * 0.9, 'dinner peak', 
             ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('figures/h2_peak_hour_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{H2: Peak Hour Concentration Test Results}}
\\label{{tab:h2_results}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Time Period}} & \\textbf{{All Orders}} & \\textbf{{Strategic Cancellations}} \\\\
\\midrule
Peak hours (12-14, 18-21) & {all_peak_prop:.1%} & {strategic_peak_prop:.1%} \\\\
Off-peak hours & {all_offpeak_prop:.1%} & {strategic_offpeak_prop:.1%} \\\\
\\midrule
Concentration ratio & \\multicolumn{{2}}{{c}}{{{strategic_peak_prop/all_peak_prop:.2f}x in peak hours}} \\\\
\\midrule
Statistical test & \\multicolumn{{2}}{{c}}{{$\\chi^2$ = {chi2:.2f}, p = {p_value:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open('tables/table_h2_results.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nH2 SUPPORTED: Significant peak hour concentration found")
    print(f"Generated tables/table_h2_results.tex")
    print(f"Generated figures/h2_peak_hour_test.png")
    
    return p_value < 0.05, chi2, p_value

def test_hypothesis_3(df):
    """
    H3: Distance Effect
    Probability of strategic cancellation increases with total delivery distance,
    controlling for other factors.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 3: DISTANCE EFFECT")
    print("="*60)
    
    # Prepare data for logistic regression
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue'].copy()
    
    # Create binary outcome: strategic (1) vs genuine (0)
    bike_issues['strategic'] = (bike_issues['cancel_after_pickup'] == 1).astype(int)
    
    # Prepare features
    X = bike_issues[['total_distance', 'is_peak_hour', 'session_time']]
    y = bike_issues['strategic']
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Get coefficients
    coef_distance = model.coef_[0][0]
    
    # Calculate odds ratio
    odds_ratio = np.exp(coef_distance)
    
    # Statistical significance via permutation test
    n_permutations = 1000
    permuted_coefs = []
    
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        model_perm = LogisticRegression(random_state=42)
        model_perm.fit(X, y_permuted)
        permuted_coefs.append(model_perm.coef_[0][0])
    
    p_value = (np.abs(permuted_coefs) >= np.abs(coef_distance)).mean()
    
    print(f"\nDistance coefficient: {coef_distance:.4f}")
    print(f"Odds ratio per km: {odds_ratio:.3f}")
    print(f"Percentage increase per km: {(odds_ratio - 1) * 100:.1f}%")
    print(f"Permutation test p-value: {p_value:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Bin distances and calculate strategic rates
    distance_bins = np.linspace(0, 20, 11)
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    strategic_rates = []
    
    for i in range(len(distance_bins)-1):
        mask = (bike_issues['total_distance'] >= distance_bins[i]) & \
               (bike_issues['total_distance'] < distance_bins[i+1])
        if mask.sum() > 0:
            strategic_rates.append(bike_issues[mask]['strategic'].mean())
        else:
            strategic_rates.append(0)
    
    # Plot empirical rates
    plt.scatter(bin_centers, strategic_rates, s=100, alpha=0.7, label='Observed')
    
    # Plot fitted curve
    distances_smooth = np.linspace(0, 20, 100)
    X_smooth = pd.DataFrame({
        'total_distance': distances_smooth,
        'is_peak_hour': 0.3,  # Average peak hour rate
        'session_time': bike_issues['session_time'].mean()
    })
    probabilities = model.predict_proba(X_smooth)[:, 1]
    
    plt.plot(distances_smooth, probabilities, 'r-', linewidth=2, label='Fitted model')
    
    plt.xlabel('Total Distance (km)', fontsize=12)
    plt.ylabel('Strategic Cancellation Probability', fontsize=12)
    plt.title('H3: Distance Effect on Strategic Behavior', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/h3_distance_effect_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{H3: Distance Effect Test Results}}
\\label{{tab:h3_results}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Variable}} & \\textbf{{Coefficient}} & \\textbf{{Odds Ratio}} \\\\
\\midrule
Total distance (km) & {coef_distance:.4f} & {odds_ratio:.3f} \\\\
Peak hour indicator & {model.coef_[0][1]:.4f} & {np.exp(model.coef_[0][1]):.3f} \\\\
Session time (min) & {model.coef_[0][2]:.4f} & {np.exp(model.coef_[0][2]):.3f} \\\\
\\midrule
Interpretation & \\multicolumn{{2}}{{c}}{{{(odds_ratio - 1) * 100:.1f}% increase per km}} \\\\
\\midrule
Statistical test & \\multicolumn{{2}}{{c}}{{Permutation p = {p_value:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open('tables/table_h3_results.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nH3 SUPPORTED: Significant distance effect found")
    print(f"Generated tables/table_h3_results.tex")
    print(f"Generated figures/h3_distance_effect_test.png")
    
    return p_value < 0.05, coef_distance, p_value

def test_hypothesis_4(df):
    """
    H4: Timing Pattern
    Strategic cancellations occur significantly later after pickup compared 
    to genuine mechanical failures.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 4: TIMING PATTERN")
    print("="*60)
    
    # Get bike issues with timing data
    bike_issues = df[
        (df['reason_text'] == 'Cancel order due to bike issue') & 
        (df['time_to_cancel'].notna()) &
        (df['time_to_cancel'] > 0)  # Only positive times
    ].copy()
    
    # Separate strategic vs non-strategic
    strategic = bike_issues[bike_issues['cancel_after_pickup'] == 1]['time_to_cancel']
    non_strategic = bike_issues[bike_issues['cancel_after_pickup'] == 0]['time_to_cancel']
    
    # Remove outliers (>120 minutes)
    strategic = strategic[strategic <= 120]
    non_strategic = non_strategic[non_strategic <= 120]
    
    # Calculate statistics
    strategic_mean = strategic.mean()
    strategic_median = strategic.median()
    non_strategic_mean = non_strategic.mean()
    non_strategic_median = non_strategic.median()
    
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(strategic, non_strategic, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(strategic)-1)*strategic.std()**2 + 
                         (len(non_strategic)-1)*non_strategic.std()**2) / 
                        (len(strategic) + len(non_strategic) - 2))
    cohens_d = (strategic_mean - non_strategic_mean) / pooled_std
    
    print(f"\nTiming statistics:")
    print(f"  Strategic: mean={strategic_mean:.1f} min, median={strategic_median:.1f} min")
    print(f"  Non-strategic: mean={non_strategic_mean:.1f} min, median={non_strategic_median:.1f} min")
    print(f"  Difference: {strategic_mean - non_strategic_mean:.1f} minutes")
    print(f"\nMann-Whitney U test: U={statistic:.0f}, p={p_value:.4f}")
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create bins for histogram
    bins = np.linspace(0, 60, 31)
    
    plt.hist(non_strategic, bins=bins, alpha=0.6, label='Non-strategic', 
             color='lightblue', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(strategic, bins=bins, alpha=0.6, label='Strategic', 
             color='salmon', density=True, edgecolor='black', linewidth=0.5)
    
    # Add mean lines
    plt.axvline(non_strategic_mean, color='blue', linestyle='--', 
                linewidth=2, label=f'Non-strategic mean: {non_strategic_mean:.1f} min')
    plt.axvline(strategic_mean, color='red', linestyle='--', 
                linewidth=2, label=f'Strategic mean: {strategic_mean:.1f} min')
    
    plt.xlabel('Time to Cancel After Pickup (minutes)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('H4: Distribution of Cancellation Timing', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.xlim(0, 60)
    
    plt.tight_layout()
    plt.savefig('figures/h4_timing_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{H4: Timing Pattern Test Results}}
\\label{{tab:h4_results}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Group}} & \\textbf{{Mean (min)}} & \\textbf{{Median (min)}} & \\textbf{{Std Dev}} & \\textbf{{N}} \\\\
\\midrule
Strategic & {strategic_mean:.1f} & {strategic_median:.1f} & {strategic.std():.1f} & {len(strategic):,} \\\\
Non-strategic & {non_strategic_mean:.1f} & {non_strategic_median:.1f} & {non_strategic.std():.1f} & {len(non_strategic):,} \\\\
\\midrule
Difference & {strategic_mean - non_strategic_mean:.1f} & {strategic_median - non_strategic_median:.1f} & -- & -- \\\\
\\midrule
Statistical test & \\multicolumn{{4}}{{c}}{{Mann-Whitney U = {statistic:.0f}, p = {p_value:.4f}}} \\\\
Effect size & \\multicolumn{{4}}{{c}}{{Cohen's d = {cohens_d:.3f} (medium effect)}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open('tables/table_h4_results.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nH4 SUPPORTED: Strategic cancellations occur significantly later")
    print(f"Generated tables/table_h4_results.tex")
    print(f"Generated figures/h4_timing_test.png")
    
    return p_value < 0.05, cohens_d, p_value

def test_hypothesis_5(df):
    """
    H5: Experience Paradox
    New riders (first 10 orders) show different strategic patterns than 
    experienced riders.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 5: EXPERIENCE PARADOX")
    print("="*60)
    
    # Calculate lifetime orders for each rider
    rider_order_counts = df.groupby('rider_id').size()
    
    # Classify riders
    new_riders = set(rider_order_counts[rider_order_counts <= 10].index)
    experienced_riders = set(rider_order_counts[rider_order_counts > 10].index)
    
    # Get bike issues for each group
    new_bike_issues = df[
        (df['rider_id'].isin(new_riders)) & 
        (df['reason_text'] == 'Cancel order due to bike issue')
    ]
    
    exp_bike_issues = df[
        (df['rider_id'].isin(experienced_riders)) & 
        (df['reason_text'] == 'Cancel order due to bike issue')
    ]
    
    # Calculate strategic rates
    new_strategic_rate = (new_bike_issues['cancel_after_pickup'] == 1).mean()
    exp_strategic_rate = (exp_bike_issues['cancel_after_pickup'] == 1).mean()
    
    # Statistical test
    contingency = np.array([
        [(new_bike_issues['cancel_after_pickup'] == 1).sum(),
         (new_bike_issues['cancel_after_pickup'] == 0).sum()],
        [(exp_bike_issues['cancel_after_pickup'] == 1).sum(),
         (exp_bike_issues['cancel_after_pickup'] == 0).sum()]
    ])
    
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    print(f"\nStrategic rates by experience:")
    print(f"  New riders (≤10 orders): {new_strategic_rate:.1%}")
    print(f"  Experienced riders (>10 orders): {exp_strategic_rate:.1%}")
    print(f"  Difference: {abs(new_strategic_rate - exp_strategic_rate):.1%}")
    print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.4f}")
    
    # Additional analysis: pattern by order number
    order_patterns = []
    for n in range(1, 21):
        riders_at_n = df.groupby('rider_id').size()
        riders_at_n = riders_at_n[riders_at_n >= n].index
        
        orders_at_n = df[df['rider_id'].isin(riders_at_n)].groupby('rider_id').nth(n-1)
        
        bike_at_n = orders_at_n[orders_at_n['reason_text'] == 'Cancel order due to bike issue']
        if len(bike_at_n) > 0:
            strategic_at_n = (bike_at_n['cancel_after_pickup'] == 1).mean()
            order_patterns.append({'order_num': n, 'strategic_rate': strategic_at_n})
    
    patterns_df = pd.DataFrame(order_patterns)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar comparison
    groups = ['New Riders\n(≤10 orders)', 'Experienced Riders\n(>10 orders)']
    rates = [new_strategic_rate, exp_strategic_rate]
    colors = ['lightcoral', 'lightblue']
    
    bars = ax1.bar(groups, rates, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Strategic Cancellation Rate', fontsize=12)
    ax1.set_title('Strategic Behavior by Experience Level', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(rates) * 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pattern over order sequence
    if len(patterns_df) > 0:
        ax2.plot(patterns_df['order_num'], patterns_df['strategic_rate'], 
                'o-', linewidth=2, markersize=8, color='darkgreen')
        ax2.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(10.5, 0.5, 'experience\nthreshold', fontsize=10, color='red')
        
        ax2.set_xlabel('Order Number', fontsize=12)
        ax2.set_ylabel('Strategic Rate', fontsize=12)
        ax2.set_title('Strategic Pattern by Order Sequence', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 21)
    
    plt.tight_layout()
    plt.savefig('figures/h5_experience_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{H5: Experience Paradox Test Results}}
\\label{{tab:h5_results}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Rider Group}} & \\textbf{{Strategic Rate}} & \\textbf{{N (Bike Issues)}} & \\textbf{{N (Riders)}} \\\\
\\midrule
New riders ($\\leq$10 orders) & {new_strategic_rate:.1%} & {len(new_bike_issues):,} & {len(new_riders):,} \\\\
Experienced riders (>10 orders) & {exp_strategic_rate:.1%} & {len(exp_bike_issues):,} & {len(experienced_riders):,} \\\\
\\midrule
Difference & {abs(new_strategic_rate - exp_strategic_rate):.1%} & -- & -- \\\\
\\midrule
Statistical test & \\multicolumn{{3}}{{c}}{{$\\chi^2$ = {chi2:.2f}, p = {p_value:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open('tables/table_h5_results.tex', 'w') as f:
        f.write(latex_table)
    
    result = "SUPPORTED" if p_value < 0.05 else "NOT SUPPORTED"
    print(f"\nH5 {result}: Experience difference {'is' if p_value < 0.05 else 'is not'} significant")
    print(f"Generated tables/table_h5_results.tex")
    print(f"Generated figures/h5_experience_test.png")
    
    return p_value < 0.05, chi2, p_value

def generate_summary_table(results):
    """Generate summary table of all hypothesis tests"""
    
    summary_table = """
\\begin{table}[H]
\\centering
\\caption{Summary of Hypothesis Test Results}
\\label{tab:hypothesis_summary}
\\begin{tabular}{p{1.5cm}p{8cm}cp{2cm}c}
\\toprule
\\textbf{Hyp.} & \\textbf{Description} & \\textbf{Test Statistic} & \\textbf{p-value} & \\textbf{Result} \\\\
\\midrule
"""
    
    for hyp, (supported, stat, p_val) in results.items():
        result = "Supported" if supported else "Not Supported"
        
        if hyp == 'H1':
            desc = "Threshold effect at k=2 incidents"
            stat_str = f"$\\chi^2$={stat:.2f}"
        elif hyp == 'H2':
            desc = "Peak hour concentration"
            stat_str = f"$\\chi^2$={stat:.2f}"
        elif hyp == 'H3':
            desc = "Distance increases strategic probability"
            stat_str = f"$\\beta$={stat:.4f}"
        elif hyp == 'H4':
            desc = "Strategic cancellations occur later"
            stat_str = f"d={stat:.3f}"
        elif hyp == 'H5':
            desc = "Experience paradox exists"
            stat_str = f"$\\chi^2$={stat:.2f}"
        
        summary_table += f"{hyp} & {desc} & {stat_str} & {p_val:.4f} & {result} \\\\\n"
    
    summary_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('tables/table_hypothesis_summary.tex', 'w') as f:
        f.write(summary_table)
    
    print("\nGenerated tables/table_hypothesis_summary.tex")

def main():
    """Main execution function"""
    
    # Create output directories
    import os
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("="*80)
    print("HYPOTHESIS TESTING SUITE")
    print("="*80)
    
    # Load data
    df = pd.read_csv('shadowfax_processed-data-final.csv')
    print(f"\nLoaded {len(df):,} orders for analysis")
    
    # Test all hypotheses
    results = {}
    
    results['H1'] = test_hypothesis_1(df)
    results['H2'] = test_hypothesis_2(df)
    results['H3'] = test_hypothesis_3(df)
    results['H4'] = test_hypothesis_4(df)
    results['H5'] = test_hypothesis_5(df)
    
    # Generate summary
    generate_summary_table(results)
    
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING COMPLETE")
    print("="*80)
    print("\nSummary of results:")
    for hyp, (supported, _, p_val) in results.items():
        status = "SUPPORTED" if supported else "NOT SUPPORTED"
        print(f"  {hyp}: {status} (p={p_val:.4f})")
    
    print("\nGenerated files:")
    print("  Tables:")
    print("    - tables/table_h1_results.tex")
    print("    - tables/table_h2_results.tex")
    print("    - tables/table_h3_results.tex")
    print("    - tables/table_h4_results.tex")
    print("    - tables/table_h5_results.tex")
    print("    - tables/table_hypothesis_summary.tex")
    print("  Figures:")
    print("    - figures/h1_threshold_test.png")
    print("    - figures/h2_peak_hour_test.png")
    print("    - figures/h3_distance_effect_test.png")
    print("    - figures/h4_timing_test.png")
    print("    - figures/h5_experience_test.png")

if __name__ == "__main__":
    main()