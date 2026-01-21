# Export Predictions Framework

A framework for predicting company revenue using export data as leading indicators. Currently implements Taiwan exports → Nvidia data center revenue prediction.

## Project Structure

```
export_predictions_contour/
├── data/
│   ├── raw/              # Raw data files (CSV)
│   └── processed/        # Cleaned and aggregated data
├── src/                  # Source code
│   ├── fetch_taiwan_exports.py    # Fetch data from Taiwan website
│   ├── clean_data.py              # Process Taiwan exports CSV
│   ├── process_nvidia_data.py     # Convert Nvidia quarterly to monthly
│   ├── predict_nvidia_revenue.py  # Prediction model
│   └── visualize_predictions.py   # Generate charts
├── models/               # Trained model files
├── notebooks/            # Generated visualizations
├── config/               # Configuration files
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Taiwan exports data:**
   
   **Option A: Automated fetch (recommended)**
   ```bash
   python src/fetch_taiwan_exports.py
   ```
   - Opens browser automatically
   - Fills form fields
   - You solve CAPTCHA manually
   - Downloads CSV automatically
   - Then run: `python src/clean_data.py`
   
   **Option B: Manual download**
   - Visit: https://portal.sw.nat.gov.tw/APGA/GA30E
   - Fill form:
     - Total Exports
     - Monthly
     - Commodity codes: `8471100000,8471200000,8471410000,8471490000,8471800000`
     - World
     - USD
     - CSV export
   - Download and save to: `data/raw/taiwan_exports_history.csv`
   - Run: `python src/clean_data.py`

3. **Process Nvidia data (if needed):**
   ```bash
   python src/process_nvidia_data.py
   ```
   This converts quarterly Nvidia revenue to monthly by dividing each quarter by 3.

4. **Train model and generate predictions:**
   ```bash
   # Default (all data)
   python src/predict_nvidia_revenue.py
   
   # Post-surge only (2023+)
   python src/predict_nvidia_revenue.py --post-surge-only
   ```

5. **Generate visualizations:**
   ```bash
   python src/visualize_predictions.py
   ```

## Current Implementation

### Data Processing

**Taiwan Exports:**
- Aggregates 5 commodity codes per month
- Converts dates to datetime format
- Creates monthly totals in USD (millions)
- Output: `data/processed/taiwan_exports_monthly.csv`

**Nvidia Revenue:**
- Converts quarterly data to monthly (quarterly revenue / 3)
- Handles fiscal year quarters
- Output: `data/processed/nvidia_data_center_revenue_monthly.csv`

### Prediction Model

**Features:**
- Current month exports
- Lag features (1, 2, 3 months)
- Rolling averages (3, 6, 12 months)
- Year-over-year and month-over-month growth rates
- Time-based features (year, month, quarter, trend)

**Model:** Ridge Regression (configurable)

**Evaluation:** RMSE, R², MAE

**Output:**
- Trained model: `models/nvidia_revenue_predictor.pkl`
- Predictions: `data/processed/nvidia_revenue_predictions.csv`

### Visualizations

Generates 5 charts in `notebooks/`:
1. **taiwan_exports_history.png** - Taiwan exports over time
2. **nvidia_actual_vs_predicted.png** - Actual vs predicted revenue
3. **taiwan_exports_vs_nvidia_revenue.png** - Combined dual-axis chart (best for presentations)
4. **correlation_scatter.png** - Correlation scatter plot
5. **prediction_accuracy.png** - Prediction accuracy analysis

## Key Insights

- **Strong Correlation:** ~0.92 correlation between Taiwan exports and Nvidia revenue
- **Regime Change:** Significant surge starting in 2023 (AI/data center boom)
- **Time Lag:** Exports typically lead revenue by 1-3 months (supply chain)
- **Scale:** Recent exports ($3-7B/month) correlate with Nvidia revenue ($10-17B/month)

## Model Performance Notes

- **Train R²:** ~0.96-0.98 (good fit on training data)
- **Test R²:** Negative (overfitting due to regime change)
- **Recommendation:** Use `--post-surge-only` flag to train only on 2023+ data for more consistent regime

## Output Files

**Data:**
- `data/processed/taiwan_exports_monthly.csv` - Cleaned Taiwan exports
- `data/processed/nvidia_data_center_revenue_monthly.csv` - Monthly Nvidia revenue
- `data/processed/nvidia_revenue_predictions.csv` - Model predictions

**Models:**
- `models/nvidia_revenue_predictor.pkl` - Trained model

**Visualizations:**
- `notebooks/taiwan_exports_vs_nvidia_revenue.png` - Main chart for presentations

## Expanding the Framework

### Adding New Data Sources
1. Add raw data to `data/raw/`
2. Create cleaning script in `src/` (follow `clean_data.py` pattern)
3. Update prediction scripts to use new data

### Adding New Prediction Targets
1. Create new prediction script (follow `predict_nvidia_revenue.py` pattern)
2. Use existing feature engineering functions
3. Train and evaluate model

### Model Improvements
- Try non-linear models (XGBoost, Random Forest)
- Use time series models (ARIMA, Prophet)
- Implement walk-forward validation
- Handle regime changes more explicitly

## Notes

- Taiwan exports data: USD thousands → converted to millions
- Nvidia data: Quarterly → monthly (divided by 3)
- Date format: YYYY/M → converted to datetime
- Commodity codes represent data processing equipment exports
- Consider time lags: exports lead revenue by 1-3 months
