#!/usr/bin/env python3
"""
00_prepare_data.py
Prepare data with all necessary features for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime

def prepare_features(df):
    """Add all necessary features to the dataset"""
    
    print("Adding time-based features...")
    
    # Parse timestamps
    time_cols = ['order_time', 'allot_time', 'accept_time', 'pickup_time', 
                 'delivered_time', 'cancelled_time']
    
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract hour from order_time
    df['hour'] = df['order_time'].dt.hour
    
    # Define peak hours (12-14, 18-21)
    peak_hours = [12, 13, 14, 18, 19, 20, 21]
    df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)
    
    # Calculate time intervals
    df['time_to_accept'] = (df['accept_time'] - df['allot_time']).dt.total_seconds() / 60
    df['time_to_pickup'] = (df['pickup_time'] - df['accept_time']).dt.total_seconds() / 60
    
    # Time to cancel (from order time)
    df['time_to_cancel'] = np.nan
    cancelled_mask = df['cancelled'] == 1
    df.loc[cancelled_mask, 'time_to_cancel'] = (
        df.loc[cancelled_mask, 'cancelled_time'] - 
        df.loc[cancelled_mask, 'order_time']
    ).dt.total_seconds() / 60
    
    # Fill NaN values for time features
    df['time_to_accept'] = df['time_to_accept'].fillna(df['time_to_accept'].median())
    df['time_to_pickup'] = df['time_to_pickup'].fillna(df['time_to_pickup'].median())
    df['time_to_cancel'] = df['time_to_cancel'].fillna(df['time_to_cancel'].median())
    
    print("Calculating rider-level metrics...")
    
    # Calculate rider-level statistics
    rider_stats = df.groupby('rider_id').agg({
        'cancelled': ['sum', 'mean'],
        'order_id': 'count'
    })
    rider_stats.columns = ['total_cancellations', 'cancel_rate', 'total_orders']
    
    # Calculate bike issue metrics
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue']
    bike_issue_stats = bike_issues.groupby('rider_id').agg({
        'order_id': 'count',
        'cancel_after_pickup': 'mean'
    })
    bike_issue_stats.columns = ['bike_issue_count', 'bike_issue_post_pickup_rate']
    
    # Merge rider stats back
    df = df.merge(rider_stats[['cancel_rate']], left_on='rider_id', right_index=True, 
                  how='left', suffixes=('', '_rider'))
    
    # Add bike issue rate
    df = df.merge(bike_issue_stats[['bike_issue_count']], left_on='rider_id', 
                  right_index=True, how='left')
    df['bike_issue_count'] = df['bike_issue_count'].fillna(0)
    df['bike_issue_rate'] = df['bike_issue_count'] / df['lifetime_order_count']
    
    # Calculate cancel after pickup ratio
    cancelled_orders = df[df['cancelled'] == 1]
    cancel_after_pickup_stats = cancelled_orders.groupby('rider_id')['cancel_after_pickup'].mean()
    df = df.merge(cancel_after_pickup_stats.rename('cancel_after_pickup_ratio'), 
                  left_on='rider_id', right_index=True, how='left')
    df['cancel_after_pickup_ratio'] = df['cancel_after_pickup_ratio'].fillna(0)
    
    # Add day of week
    df['day_of_week'] = df['order_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    print(f"Total features added: {len(df.columns) - 21}")
    
    return df

def validate_data(df):
    """Validate the prepared dataset"""
    
    print("\nData validation:")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check for missing values in key columns
    key_cols = ['hour', 'is_peak_hour', 'time_to_accept', 'time_to_pickup',
                'cancel_rate', 'bike_issue_rate']
    
    print("\nMissing values in key columns:")
    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
    
    # Basic statistics
    print("\nCancellation statistics:")
    print(f"  Overall cancellation rate: {df['cancelled'].mean():.2%}")
    print(f"  Bike issue cancellations: {len(df[df['reason_text'] == 'Cancel order due to bike issue']):,}")
    print(f"  Post-pickup cancellation rate: {df['cancel_after_pickup'].mean():.2%}")

def main():
    """Main execution function"""
    
    print("="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('shadowfax_processed-data-final.csv')
    print(f"Loaded {len(df):,} rows")
    
    # Prepare features
    df_prepared = prepare_features(df)
    
    # Validate
    validate_data(df_prepared)
    
    # Save prepared data
    output_file = 'train_prepared.csv'
    df_prepared.to_csv(output_file, index=False)
    print(f"\nSaved prepared data to {output_file}")
    
    # Show final columns
    print("\nFinal columns:")
    for i, col in enumerate(df_prepared.columns, 1):
        print(f"  {i:2d}. {col}")

if __name__ == "__main__":
    main()