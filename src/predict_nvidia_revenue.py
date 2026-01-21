"""
Predict Nvidia data center revenue using Taiwan exports data.

This script:
1. Loads cleaned Taiwan exports data
2. Fetches/loads Nvidia data center revenue data (if available)
3. Trains a model to predict Nvidia revenue from Taiwan exports
4. Generates predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def load_taiwan_exports():
    """Load cleaned Taiwan exports data."""
    file_path = PROCESSED_DATA_DIR / "taiwan_exports_monthly.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {file_path}\n"
            "Please run clean_data.py first."
        )
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def fetch_nvidia_data():
    """
    Load Nvidia data center revenue data from processed CSV.
    
    The data is monthly, converted from quarterly reports where
    each quarter's revenue is evenly divided across its 3 months.
    """
    file_path = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_monthly.csv"
    
    if not file_path.exists():
        print(f"⚠️  Nvidia data file not found: {file_path}")
        print("   Please run src/process_nvidia_data.py first to generate the monthly data.")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Return in the expected format
    return {
        'date': df['date'].values,
        'data_center_revenue_millions': df['data_center_revenue_millions'].values
    }

def create_features(df):
    """
    Create features from Taiwan exports data.
    Includes lag features, rolling averages, etc.
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Lag features (previous months)
    df['exports_lag1'] = df['monthly_exports_usd_millions'].shift(1)
    df['exports_lag2'] = df['monthly_exports_usd_millions'].shift(2)
    df['exports_lag3'] = df['monthly_exports_usd_millions'].shift(3)
    
    # Rolling averages
    df['exports_ma3'] = df['monthly_exports_usd_millions'].rolling(window=3).mean()
    df['exports_ma6'] = df['monthly_exports_usd_millions'].rolling(window=6).mean()
    df['exports_ma12'] = df['monthly_exports_usd_millions'].rolling(window=12).mean()
    
    # Year-over-year growth
    df['exports_yoy'] = df['monthly_exports_usd_millions'].pct_change(periods=12)
    
    # Month-over-month growth
    df['exports_mom'] = df['monthly_exports_usd_millions'].pct_change(periods=1)
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Trend (linear time trend)
    df['trend'] = range(len(df))
    
    return df

def train_model(X_train, y_train, X_test, y_test, model_type='ridge'):
    """
    Train a regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: 'linear' or 'ridge'
    
    Returns:
        Trained model and metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Choose model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = Ridge(alpha=1.0)
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'scaler': scaler
    }
    
    return model, metrics

def plot_results(df, y_actual, y_pred, dates, title="Predictions vs Actual"):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_actual, label='Actual', marker='o', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted', marker='x', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Revenue (Millions USD)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plots_dir = PROJECT_ROOT / "notebooks"
    plots_dir.mkdir(exist_ok=True)
    filename = title.lower().replace(' ', '_') + '.png'
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=150)
    print(f"Plot saved to {filepath}")
    plt.close()

def main(use_post_surge_only=False, surge_threshold_date='2023-01-01'):
    """
    Main prediction pipeline.
    
    Args:
        use_post_surge_only: If True, only use data from surge period onwards for training
        surge_threshold_date: Date to consider as start of surge period (default: '2023-01-01')
    
    Note: This assumes you have Nvidia data center revenue data.
    If not available, the script will demonstrate the feature engineering
    and model structure that would be used.
    """
    print("=" * 60)
    print("Nvidia Data Center Revenue Prediction")
    print("=" * 60)
    
    # Load Taiwan exports data
    print("\n1. Loading Taiwan exports data...")
    taiwan_df = load_taiwan_exports()
    print(f"   Loaded {len(taiwan_df)} months of data")
    
    # Create features
    print("\n2. Creating features...")
    taiwan_df = create_features(taiwan_df)
    
    # Drop rows with NaN (from lag features)
    taiwan_df = taiwan_df.dropna().reset_index(drop=True)
    
    # Feature columns (excluding date and target-related columns)
    feature_cols = [
        'monthly_exports_usd_millions',
        'exports_lag1', 'exports_lag2', 'exports_lag3',
        'exports_ma3', 'exports_ma6', 'exports_ma12',
        'exports_yoy', 'exports_mom',
        'year', 'month', 'quarter', 'trend'
    ]
    
    # Check which features are available (some might be NaN)
    available_features = [col for col in feature_cols if col in taiwan_df.columns]
    X = taiwan_df[available_features].fillna(0)
    
    print(f"   Created {len(available_features)} features")
    print(f"   Features: {', '.join(available_features)}")
    
    # Try to load Nvidia data
    print("\n3. Loading Nvidia data center revenue...")
    nvidia_data = fetch_nvidia_data()
    
    if nvidia_data is None:
        print("   ⚠️  Nvidia data not available.")
        print("   This script demonstrates the prediction framework.")
        print("   To use it, you need to:")
        print("   1. Obtain Nvidia quarterly/monthly data center revenue")
        print("   2. Align it with Taiwan exports data by date")
        print("   3. Update the fetch_nvidia_data() function")
        print("\n   For now, showing Taiwan exports data structure:")
        print(taiwan_df[['date', 'monthly_exports_usd_millions']].tail(12))
        return
    
    # Merge Nvidia data with Taiwan exports
    nvidia_df = pd.DataFrame(nvidia_data)
    nvidia_df['date'] = pd.to_datetime(nvidia_df['date'])
    
    # Filter Taiwan exports to only include dates where we have Nvidia data
    # This prevents using data before we can validate the relationship
    first_nvidia_date = nvidia_df['date'].min()
    last_nvidia_date = nvidia_df['date'].max()
    
    print(f"   Nvidia data range: {first_nvidia_date.date()} to {last_nvidia_date.date()}")
    print(f"   Filtering Taiwan exports to this date range...")
    
    taiwan_df_filtered = taiwan_df[
        (taiwan_df['date'] >= first_nvidia_date) & 
        (taiwan_df['date'] <= last_nvidia_date)
    ].copy()
    
    print(f"   Taiwan exports after filtering: {len(taiwan_df_filtered)} months")
    
    # Recreate features on filtered data (to ensure proper lag calculations)
    taiwan_df_filtered = create_features(taiwan_df_filtered)
    taiwan_df_filtered = taiwan_df_filtered.dropna().reset_index(drop=True)
    
    # Merge on date
    merged_df = pd.merge(
        taiwan_df_filtered,
        nvidia_df,
        on='date',
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("   ⚠️  No overlapping dates found between Taiwan exports and Nvidia data")
        return
    
    # Sort by date to ensure chronological order
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    print(f"   Final merged dataset: {len(merged_df)} months")
    print(f"   Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    
    # Option to filter to post-surge period only
    surge_threshold = pd.Timestamp(surge_threshold_date)
    if use_post_surge_only:
        print(f"\n   Filtering to post-surge period only (>= {surge_threshold.date()})...")
        merged_df = merged_df[merged_df['date'] >= surge_threshold].reset_index(drop=True)
        print(f"   Dataset after filtering: {len(merged_df)} months")
        print(f"   Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    
    # Prepare target variable
    y = merged_df['data_center_revenue_millions'].values
    X_merged = merged_df[available_features].fillna(0)
    dates = merged_df['date'].values
    
    # Time-based split: train on first 80% chronologically, test on last 20%
    # This is more appropriate for time series and avoids data leakage
    print(f"\n4. Splitting data chronologically (n={len(merged_df)})...")
    split_idx = int(len(merged_df) * 0.8)
    
    X_train = X_merged.iloc[:split_idx]
    X_test = X_merged.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    print(f"   Train: {len(X_train)} samples ({train_dates[0]} to {train_dates[-1]})")
    print(f"   Test: {len(X_test)} samples ({test_dates[0]} to {test_dates[-1]})")
    
    # Train model
    print("\n5. Training model...")
    model, metrics = train_model(X_train, y_train, X_test, y_test, model_type='ridge')
    
    # Print metrics
    print("\n6. Model Performance:")
    print(f"   Train RMSE: ${metrics['train_rmse']:.2f}M")
    print(f"   Test RMSE: ${metrics['test_rmse']:.2f}M")
    print(f"   Train R²: {metrics['train_r2']:.4f}")
    print(f"   Test R²: {metrics['test_r2']:.4f}")
    print(f"   Train MAE: ${metrics['train_mae']:.2f}M")
    print(f"   Test MAE: ${metrics['test_mae']:.2f}M")
    
    # Additional analysis: check for regime change
    print("\n   Regime Analysis:")
    surge_threshold_ts = pd.Timestamp(surge_threshold_date)
    train_pre_surge = len([d for d in train_dates if pd.Timestamp(d) < surge_threshold_ts])
    train_post_surge = len([d for d in train_dates if pd.Timestamp(d) >= surge_threshold_ts])
    test_pre_surge = len([d for d in test_dates if pd.Timestamp(d) < surge_threshold_ts])
    test_post_surge = len([d for d in test_dates if pd.Timestamp(d) >= surge_threshold_ts])
    
    print(f"   Train: {train_pre_surge} pre-surge, {train_post_surge} post-surge months")
    print(f"   Test: {test_pre_surge} pre-surge, {test_post_surge} post-surge months")
    
    if test_post_surge > 0 and train_pre_surge > train_post_surge and not use_post_surge_only:
        print(f"   ⚠️  Warning: Model trained mostly on pre-surge data but tested on post-surge data")
        print(f"      This may explain poor test performance.")
        print(f"      Consider using use_post_surge_only=True to train only on post-surge data.")
    
    # Save model
    import joblib
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "nvidia_revenue_predictor.pkl"
    joblib.dump({
        'model': model,
        'scaler': metrics['scaler'],
        'features': available_features
    }, model_path)
    print(f"\n7. Model saved to: {model_path}")
    
    # Generate predictions for recent months
    print("\n8. Generating predictions for recent months...")
    # Use filtered Taiwan data for predictions to ensure consistency
    recent_data = taiwan_df_filtered.tail(12)
    X_recent = recent_data[available_features].fillna(0)
    X_recent_scaled = metrics['scaler'].transform(X_recent)
    predictions = model.predict(X_recent_scaled)
    
    predictions_df = pd.DataFrame({
        'date': recent_data['date'].values,
        'taiwan_exports_millions': recent_data['monthly_exports_usd_millions'].values,
        'predicted_nvidia_revenue_millions': predictions
    })
    
    print("\nRecent Predictions:")
    print(predictions_df.to_string(index=False))
    
    # Save predictions
    predictions_path = PROCESSED_DATA_DIR / "nvidia_revenue_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")
    
    # Generate visualizations
    print("\n9. Generating visualizations...")
    try:
        import sys
        import importlib.util
        viz_path = Path(__file__).parent / "visualize_predictions.py"
        if viz_path.exists():
            spec = importlib.util.spec_from_file_location("visualize_predictions", viz_path)
            viz_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(viz_module)
            viz_module.main()
        else:
            print("   ⚠️  Visualization script not found. Run src/visualize_predictions.py separately.")
    except Exception as e:
        print(f"   ⚠️  Error generating visualizations: {e}")
        print("   You can run src/visualize_predictions.py separately to generate plots.")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument to use post-surge only
    use_post_surge = '--post-surge-only' in sys.argv or '--post-surge' in sys.argv
    
    if use_post_surge:
        print("=" * 60)
        print("Training on POST-SURGE data only (>= 2023-01-01)")
        print("=" * 60)
    
    main(use_post_surge_only=use_post_surge)
