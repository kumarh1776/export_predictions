"""Predict Nvidia data center revenue using Taiwan exports data."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def load_taiwan_exports():
    """Load cleaned Taiwan exports monthly data."""
    file_path = PROCESSED_DATA_DIR / "taiwan_exports_monthly.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {file_path}. Run clean_data.py first.")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def fetch_nvidia_data():
    """Load Nvidia data center revenue (monthly, converted from quarterly reports)."""
    file_path = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_monthly.csv"
    if not file_path.exists():
        print(f"Nvidia data not found: {file_path}. Run process_nvidia_data.py first.")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return {'date': df['date'].values, 'data_center_revenue_millions': df['data_center_revenue_millions'].values}

def create_features(df):
    """Create time series features: lags, rolling averages, growth rates, and time indicators."""
    df = df.copy().sort_values('date').reset_index(drop=True)
    
    # Lag features (previous months)
    df['exports_lag1'] = df['monthly_exports_usd_millions'].shift(1)
    df['exports_lag2'] = df['monthly_exports_usd_millions'].shift(2)
    df['exports_lag3'] = df['monthly_exports_usd_millions'].shift(3)
    
    # Rolling averages
    df['exports_ma3'] = df['monthly_exports_usd_millions'].rolling(window=3).mean()
    df['exports_ma6'] = df['monthly_exports_usd_millions'].rolling(window=6).mean()
    df['exports_ma12'] = df['monthly_exports_usd_millions'].rolling(window=12).mean()
    
    # Growth rates
    df['exports_yoy'] = df['monthly_exports_usd_millions'].pct_change(periods=12)
    df['exports_mom'] = df['monthly_exports_usd_millions'].pct_change(periods=1)
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    return df

def train_model(X_train, y_train, X_test, y_test, use_cv=True):
    """Train Ridge regression model with cross-validation for optimal regularization."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use cross-validation to find optimal alpha, or use fixed value
    if use_cv:
        alphas = np.logspace(-2, 3, 50)  # Range from 0.01 to 1000
        tscv = TimeSeriesSplit(n_splits=min(5, len(X_train) // 3))
        model = RidgeCV(alphas=alphas, cv=tscv, scoring='r2')
    else:
        model = Ridge(alpha=10.0)
    
    model.fit(X_train_scaled, y_train)
    
    if use_cv and hasattr(model, 'alpha_'):
        print(f"Selected Ridge alpha: {model.alpha_:.4f}")
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Baseline: predict mean of training data
    baseline_r2 = r2_score(y_test, np.full_like(y_test, y_train.mean()))
    
    return model, {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'baseline_r2': baseline_r2,
        'scaler': scaler,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def analyze_feature_importance(model, X_train, X_test, y_train, y_test, feature_names, scaler):
    """Analyze feature importance using three methods: coefficients, permutation, and correlation."""
    # Method 1: Coefficient magnitude (model-specific)
    coefficients = model.coef_
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Method 2: Permutation importance (model-agnostic, measures impact of shuffling)
    X_test_scaled = scaler.transform(X_test)
    perm_importance = permutation_importance(
        model, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='r2'
    )
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std
    })
    
    # Method 3: Correlation with target (simple baseline)
    X_combined = pd.concat([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    correlations = [abs(np.corrcoef(X_combined[f].values, y_combined)[0, 1]) 
                    if not np.isnan(np.corrcoef(X_combined[f].values, y_combined)[0, 1]) else 0
                    for f in feature_names]
    corr_df = pd.DataFrame({
        'feature': feature_names,
        'abs_correlation': correlations
    })
    
    # Combine all methods and normalize to 0-1 range
    importance_df = coef_df.merge(perm_df, on='feature').merge(corr_df, on='feature')
    
    importance_df['coef_score_norm'] = importance_df['abs_coefficient'] / importance_df['abs_coefficient'].max()
    importance_df['perm_score_norm'] = importance_df['perm_importance_mean'] / importance_df['perm_importance_mean'].max()
    importance_df['corr_score_norm'] = importance_df['abs_correlation'] / importance_df['abs_correlation'].max()
    
    # Combined score: average of all three normalized scores
    importance_df['combined_score'] = (importance_df['coef_score_norm'] + 
                                       importance_df['perm_score_norm'] + 
                                       importance_df['corr_score_norm']) / 3
    
    return {'importance_df': importance_df.sort_values('combined_score', ascending=False)}

def plot_feature_importance(importance_df):
    """Create 2x2 subplot showing feature importance from different methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    df_sorted = importance_df.sort_values('combined_score', ascending=True)
    
    axes[0, 0].barh(df_sorted['feature'], df_sorted['abs_coefficient'])
    axes[0, 0].set_xlabel('Absolute Coefficient')
    axes[0, 0].set_title('Coefficient Magnitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].barh(df_sorted['feature'], df_sorted['perm_importance_mean'], 
                    xerr=df_sorted['perm_importance_std'], capsize=3)
    axes[0, 1].set_xlabel('Permutation Importance')
    axes[0, 1].set_title('Permutation Importance')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].barh(df_sorted['feature'], df_sorted['abs_correlation'])
    axes[1, 0].set_xlabel('Absolute Correlation')
    axes[1, 0].set_title('Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].barh(df_sorted['feature'], df_sorted['combined_score'], color='steelblue')
    axes[1, 1].set_xlabel('Combined Score')
    axes[1, 1].set_title('Combined Importance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots_dir = PROJECT_ROOT / "notebooks"
    plots_dir.mkdir(exist_ok=True)
    filepath = plots_dir / "feature_importance.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {filepath}")

def plot_train_test_comparison(train_dates, y_train, y_train_pred, test_dates, y_test, y_test_pred):
    """Plot train vs test predictions: time series view and scatter plot."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: time series showing train/test split
    axes[0].plot(train_dates, y_train, 'o-', label='Train Actual', alpha=0.7, color='blue')
    axes[0].plot(train_dates, y_train_pred, 'x-', label='Train Predicted', alpha=0.7, color='lightblue')
    axes[0].plot(test_dates, y_test, 'o-', label='Test Actual', alpha=0.7, color='red')
    axes[0].plot(test_dates, y_test_pred, 'x-', label='Test Predicted', alpha=0.7, color='salmon')
    axes[0].axvline(x=test_dates[0], color='gray', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Revenue (Millions USD)')
    axes[0].set_title('Train vs Test Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Bottom plot: scatter plot of actual vs predicted
    min_val = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
    max_val = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
    
    axes[1].scatter(y_train, y_train_pred, alpha=0.6, 
                    label=f'Train (R²={r2_score(y_train, y_train_pred):.3f})', color='blue')
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, 
                    label=f'Test (R²={r2_score(y_test, y_test_pred):.3f})', color='red')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
    axes[1].set_xlabel('Actual Revenue (Millions USD)')
    axes[1].set_ylabel('Predicted Revenue (Millions USD)')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots_dir = PROJECT_ROOT / "notebooks"
    plots_dir.mkdir(exist_ok=True)
    filepath = plots_dir / "train_test_diagnostics.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plot saved to {filepath}")

def main(use_post_surge_only=False, surge_threshold_date='2023-01-01'):
    """Main prediction pipeline."""
    print("Nvidia Data Center Revenue Prediction")
    print("=" * 60)
    
    # Load and prepare Taiwan exports data
    taiwan_df = load_taiwan_exports()
    print(f"Loaded {len(taiwan_df)} months of Taiwan exports data")
    
    taiwan_df = create_features(taiwan_df)
    taiwan_df = taiwan_df.dropna().reset_index(drop=True)
    
    feature_cols = [
        'monthly_exports_usd_millions', 'exports_lag1', 'exports_lag2', 'exports_lag3',
        'exports_ma3', 'exports_ma6', 'exports_ma12', 'exports_yoy', 'exports_mom',
        'year', 'month', 'quarter'
    ]
    available_features = [col for col in feature_cols if col in taiwan_df.columns]
    
    print(f"Created {len(available_features)} features")
    
    # Load Nvidia revenue data
    nvidia_data = fetch_nvidia_data()
    if nvidia_data is None:
        print("Nvidia data not available. Run process_nvidia_data.py first.")
        return
    
    nvidia_df = pd.DataFrame(nvidia_data)
    nvidia_df['date'] = pd.to_datetime(nvidia_df['date'])
    
    # Filter Taiwan data to match Nvidia date range
    first_nvidia_date = nvidia_df['date'].min()
    last_nvidia_date = nvidia_df['date'].max()
    
    taiwan_df_filtered = taiwan_df[
        (taiwan_df['date'] >= first_nvidia_date) & 
        (taiwan_df['date'] <= last_nvidia_date)
    ].copy()
    
    # Recreate features on filtered data (ensures proper lag calculations)
    taiwan_df_filtered = create_features(taiwan_df_filtered)
    taiwan_df_filtered = taiwan_df_filtered.dropna().reset_index(drop=True)
    
    # Merge datasets on date
    merged_df = pd.merge(taiwan_df_filtered, nvidia_df, on='date', how='inner')
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    if len(merged_df) == 0:
        print("No overlapping dates found")
        return
    
    print(f"Merged dataset: {len(merged_df)} months")
    print(f"Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    
    # Optional: filter to post-surge period only (2023+)
    surge_threshold = pd.Timestamp(surge_threshold_date)
    if use_post_surge_only:
        merged_df = merged_df[merged_df['date'] >= surge_threshold].reset_index(drop=True)
        print(f"Filtered to post-surge: {len(merged_df)} months")
    
    # Prepare features and target
    y = merged_df['data_center_revenue_millions'].values
    X_merged = merged_df[available_features].fillna(0)
    dates = merged_df['date'].values
    
    # Time-based split: 80% train, 20% test (chronological)
    split_idx = int(len(merged_df) * 0.8)
    X_train = X_merged.iloc[:split_idx]
    X_test = X_merged.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples ({train_dates[0]} to {train_dates[-1]})")
    print(f"Test: {len(X_test)} samples ({test_dates[0]} to {test_dates[-1]})")
    
    # Train model
    print("\nTraining model...")
    model, metrics = train_model(X_train, y_train, X_test, y_test, use_cv=True)
    
    # Display performance metrics
    print("\nModel Performance:")
    print(f"  Train RMSE: ${metrics['train_rmse']:.2f}M | R²: {metrics['train_r2']:.4f}")
    print(f"  Test RMSE: ${metrics['test_rmse']:.2f}M | R²: {metrics['test_r2']:.4f}")
    print(f"  Baseline R²: {metrics['baseline_r2']:.4f}")
    
    # Check for regime change issues
    if metrics['test_r2'] < 0:
        train_mean = y_train.mean()
        test_mean = y_test.mean()
        print(f"\nWARNING: Negative test R² ({metrics['test_r2']:.4f})")
        print(f"Train mean: ${train_mean:.2f}M | Test mean: ${test_mean:.2f}M")
        if abs(test_mean - train_mean) > 2 * y_train.std():
            print("Significant regime change detected. Consider --post-surge-only flag.")
    
    # Generate diagnostic plots
    plot_train_test_comparison(train_dates, y_train, metrics['y_train_pred'],
                               test_dates, y_test, metrics['y_test_pred'])
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_results = analyze_feature_importance(
        model, X_train, X_test, y_train, y_test, available_features, metrics['scaler']
    )
    importance_df = importance_results['importance_df']
    
    print("\nTop 10 Features (by Combined Score):")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:25s} | "
              f"Coef: {row['abs_coefficient']:8.4f} | "
              f"Perm: {row['perm_importance_mean']:8.4f} | "
              f"Corr: {row['abs_correlation']:6.4f}")
    
    # Flag features that hurt performance
    negative_features = importance_df[importance_df['perm_importance_mean'] < -0.01]
    if len(negative_features) > 0:
        print("\nFeatures with negative importance (consider removing):")
        for idx, row in negative_features.iterrows():
            print(f"  {row['feature']:25s} | {row['perm_importance_mean']:10.4f}")
    
    # Save feature importance analysis
    importance_path = PROCESSED_DATA_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    plot_feature_importance(importance_df)
    
    # Save trained model
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "nvidia_revenue_predictor.pkl"
    joblib.dump({
        'model': model,
        'scaler': metrics['scaler'],
        'features': available_features
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Generate predictions for recent months
    recent_data = taiwan_df_filtered.tail(12)
    X_recent = recent_data[available_features].fillna(0)
    X_recent_scaled = metrics['scaler'].transform(X_recent)
    predictions = model.predict(X_recent_scaled)
    
    predictions_df = pd.DataFrame({
        'date': recent_data['date'].values,
        'taiwan_exports_millions': recent_data['monthly_exports_usd_millions'].values,
        'predicted_nvidia_revenue_millions': predictions
    })
    
    predictions_path = PROCESSED_DATA_DIR / "nvidia_revenue_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Generate additional visualizations
    try:
        import importlib.util
        viz_path = Path(__file__).parent / "visualize_predictions.py"
        if viz_path.exists():
            spec = importlib.util.spec_from_file_location("visualize_predictions", viz_path)
            viz_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(viz_module)
            viz_module.main()
    except Exception as e:
        print(f"Could not generate visualizations: {e}")

if __name__ == "__main__":
    import sys
    use_post_surge = '--post-surge-only' in sys.argv or '--post-surge' in sys.argv
    if use_post_surge:
        print("Training on POST-SURGE data only (>= 2023-01-01)")
        print("=" * 60)
    main(use_post_surge_only=use_post_surge)
