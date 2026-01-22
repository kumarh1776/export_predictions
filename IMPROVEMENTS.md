# Model Performance Improvements

## Problem: Negative Test R² (-1.11)

A negative R² score means the model performs **worse than simply predicting the mean** of the training data. This is a serious issue indicating fundamental problems.

## Root Causes

### 1. **Regime Change (Most Likely)**
- **Issue**: Training on pre-2023 data (lower revenue, ~$2-5B/month) but testing on post-2023 data (AI boom, ~$10-17B/month)
- **Impact**: Model learned patterns from a different economic regime
- **Evidence**: Large mean shift between train and test periods

### 2. **Overfitting**
- **Issue**: Model memorized training data patterns that don't generalize
- **Evidence**: High train R² (~0.96) but negative test R²
- **Contributing factors**:
  - Too many features relative to data points
  - Weak regularization (Ridge alpha=1.0 was too low)
  - Problematic features like `trend` (linear counter)

### 3. **Problematic Features**
- **`trend`**: Linear counter (0, 1, 2, ...) that doesn't generalize across regimes
- **Negative importance features**: Some features have negative permutation importance, meaning they hurt performance

## Improvements Implemented

### 1. **Better Regularization**
- ✅ **RidgeCV**: Automatic cross-validation to find optimal alpha
- ✅ **Higher default alpha**: Changed from 1.0 to 10.0 (or CV-selected)
- ✅ **Time series cross-validation**: Uses `TimeSeriesSplit` to respect temporal order

### 2. **Feature Engineering**
- ✅ **Removed `trend` feature**: This linear counter was causing overfitting
- ✅ **Feature importance analysis**: Identifies which features help/hurt
- ✅ **Negative feature detection**: Flags features with negative permutation importance

### 3. **Enhanced Diagnostics**
- ✅ **Baseline comparison**: Shows R² vs. mean baseline
- ✅ **Regime analysis**: Detects and reports mean shifts between train/test
- ✅ **Diagnostic plots**: Visual comparison of train vs test performance
- ✅ **Detailed warnings**: Explains why test R² is negative

### 4. **Better Visualization**
- ✅ **Train/test comparison plot**: Shows predictions across time
- ✅ **Scatter plots**: Separate train/test scatter plots to see overfitting

## How to Use

### Option 1: Use Post-Surge Data Only (Recommended)
```bash
python src/predict_nvidia_revenue.py --post-surge-only
```
This trains only on 2023+ data, avoiding the regime change issue.

### Option 2: Review Feature Importance
After running, check `data/processed/feature_importance.csv`:
- Remove features with negative `perm_importance_mean`
- Focus on top features by `combined_score`

### Option 3: Manual Feature Selection
1. Run the script to see which features have negative importance
2. Remove those features from `feature_cols` list
3. Re-run the script

## Expected Improvements

With these changes, you should see:
- **Better test R²**: Should be positive (ideally > 0.5)
- **Reduced overfitting**: Smaller gap between train and test R²
- **More stable predictions**: Less sensitive to regime changes

## Additional Recommendations

### 1. **Walk-Forward Validation**
Instead of single train/test split, use rolling window:
- Train on months 1-24, test on month 25
- Train on months 2-25, test on month 26
- etc.

### 2. **Feature Selection**
Manually remove features with:
- Negative permutation importance
- Very low correlation with target
- High multicollinearity

### 3. **Alternative Models**
Consider:
- **XGBoost/LightGBM**: Better at handling non-linear relationships
- **Time series models**: ARIMA, Prophet (handle trends better)
- **Ensemble methods**: Combine multiple models

### 4. **Regime-Aware Modeling**
- Train separate models for pre-2023 and post-2023
- Use regime detection to switch models
- Add regime indicator as a feature

## Monitoring

After improvements, monitor:
- **Test R²**: Should be positive and ideally > 0.5
- **Train-test gap**: Should be < 0.2 (train R² - test R²)
- **RMSE**: Should be reasonable relative to revenue scale (~$1-2B is good)
- **Feature importance**: Should be stable across different train/test splits

## Files Generated

- `notebooks/train_test_diagnostics.png`: Visual comparison of train vs test
- `data/processed/feature_importance.csv`: Detailed feature importance metrics
- `notebooks/feature_importance.png`: Feature importance visualization
