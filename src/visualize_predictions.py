"""
Visualize Taiwan exports vs Nvidia revenue predictions.

Creates charts showing:
1. Taiwan exports over time
2. Nvidia actual revenue vs predictions
3. Correlation/scatter plot
4. Combined time series
5. Trading signals (high-value for hedge funds)
6. Earnings surprise predictions (high-value for hedge funds)
7. Regime analysis (high-value for hedge funds)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

def load_data():
    """Load all necessary data files."""
    # Taiwan exports
    taiwan_file = PROCESSED_DATA_DIR / "taiwan_exports_monthly.csv"
    taiwan_df = pd.read_csv(taiwan_file)
    taiwan_df['date'] = pd.to_datetime(taiwan_df['date'])
    
    # Nvidia actual revenue
    nvidia_file = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_monthly.csv"
    nvidia_df = pd.read_csv(nvidia_file)
    nvidia_df['date'] = pd.to_datetime(nvidia_df['date'])
    
    # Nvidia predictions
    predictions_file = PROCESSED_DATA_DIR / "nvidia_revenue_predictions.csv"
    predictions_df = None
    if predictions_file.exists():
        predictions_df = pd.read_csv(predictions_file)
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    return taiwan_df, nvidia_df, predictions_df

def plot_taiwan_exports(taiwan_df):
    """Plot Taiwan exports over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(taiwan_df['date'], taiwan_df['monthly_exports_usd_millions'], 
            linewidth=2, color='#2E86AB', label='Taiwan Exports')
    
    # Highlight surge period
    surge_start = pd.Timestamp('2023-01-01')
    ax.axvspan(surge_start, taiwan_df['date'].max(), alpha=0.2, color='orange', 
               label='Surge Period (2023+)')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monthly Exports (Millions USD)', fontsize=12, fontweight='bold')
    ax.set_title('Taiwan Monthly Exports: Data Processing Equipment', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = NOTEBOOKS_DIR / "taiwan_exports_history.png"
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_nvidia_vs_predictions(nvidia_df, predictions_df):
    """Plot Nvidia actual revenue vs predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual revenue
    ax.plot(nvidia_df['date'], nvidia_df['data_center_revenue_millions'], 
            linewidth=2.5, color='#A23B72', marker='o', markersize=4,
            label='Nvidia Actual Revenue', alpha=0.8)
    
    # Plot predictions if available
    if predictions_df is not None:
        ax.plot(predictions_df['date'], predictions_df['predicted_nvidia_revenue_millions'],
                linewidth=2, color='#F18F01', marker='x', markersize=6,
                label='Nvidia Predicted Revenue', alpha=0.8, linestyle='--')
    
    # Highlight surge period
    surge_start = pd.Timestamp('2023-01-01')
    ax.axvspan(surge_start, nvidia_df['date'].max(), alpha=0.2, color='orange',
               label='Surge Period (2023+)')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Data Center Revenue (Millions USD)', fontsize=12, fontweight='bold')
    ax.set_title('Nvidia Data Center Revenue: Actual vs Predicted', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = NOTEBOOKS_DIR / "nvidia_actual_vs_predicted.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_combined_comparison(taiwan_df, nvidia_df, predictions_df):
    """Plot Taiwan exports vs Nvidia revenue on dual y-axis."""
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Merge data on date
    merged = pd.merge(taiwan_df, nvidia_df, on='date', how='inner', suffixes=('_taiwan', '_nvidia'))
    
    # Left y-axis: Taiwan exports
    color1 = '#2E86AB'
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Taiwan Exports (Millions USD)', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(merged['date'], merged['monthly_exports_usd_millions'], 
                     linewidth=2.5, color=color1, label='Taiwan Exports', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Nvidia revenue
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('Nvidia Data Center Revenue (Millions USD)', 
                   fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(merged['date'], merged['data_center_revenue_millions'],
                     linewidth=2.5, color=color2, marker='o', markersize=3,
                     label='Nvidia Actual Revenue', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add predictions if available
    if predictions_df is not None:
        # Merge predictions with actual for comparison
        pred_merged = pd.merge(predictions_df, nvidia_df, on='date', how='left', 
                               suffixes=('_pred', '_actual'))
        line3 = ax2.plot(pred_merged['date'], pred_merged['predicted_nvidia_revenue_millions'],
                         linewidth=2, color='#F18F01', marker='x', markersize=5,
                         label='Nvidia Predicted Revenue', alpha=0.8, linestyle='--')
    
    # Highlight surge period
    surge_start = pd.Timestamp('2023-01-01')
    ax1.axvspan(surge_start, merged['date'].max(), alpha=0.15, color='orange',
                label='Surge Period (2023+)')
    
    # Combine legends
    lines = line1 + line2
    if predictions_df is not None:
        lines = lines + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    ax1.set_title('Taiwan Exports vs Nvidia Data Center Revenue', 
                  fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = NOTEBOOKS_DIR / "taiwan_exports_vs_nvidia_revenue.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_correlation_scatter(taiwan_df, nvidia_df):
    """Plot scatter plot showing correlation between Taiwan exports and Nvidia revenue."""
    # Merge data
    merged = pd.merge(taiwan_df, nvidia_df, on='date', how='inner', 
                     suffixes=('_taiwan', '_nvidia'))
    
    # Calculate correlation
    correlation = merged['monthly_exports_usd_millions'].corr(merged['data_center_revenue_millions'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by date (more recent = darker)
    colors = plt.cm.viridis(np.linspace(0, 1, len(merged)))
    
    scatter = ax.scatter(merged['monthly_exports_usd_millions'], 
                        merged['data_center_revenue_millions'],
                        c=range(len(merged)), cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(merged['monthly_exports_usd_millions'], 
                   merged['data_center_revenue_millions'], 1)
    p = np.poly1d(z)
    ax.plot(merged['monthly_exports_usd_millions'], 
           p(merged['monthly_exports_usd_millions']), 
           "r--", alpha=0.8, linewidth=2, label=f'Trend Line (r={correlation:.3f})')
    
    ax.set_xlabel('Taiwan Monthly Exports (Millions USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nvidia Data Center Revenue (Millions USD)', fontsize=12, fontweight='bold')
    ax.set_title(f'Taiwan Exports vs Nvidia Revenue Correlation\n(Correlation: {correlation:.3f})',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Progression', fontsize=10)
    
    plt.tight_layout()
    filepath = NOTEBOOKS_DIR / "correlation_scatter.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()
    
    return correlation

def plot_prediction_accuracy(nvidia_df, predictions_df):
    """Plot prediction accuracy with error bars."""
    if predictions_df is None:
        print("⚠️  No predictions file found, skipping accuracy plot")
        return
    
    # Merge predictions with actual
    merged = pd.merge(predictions_df, nvidia_df, on='date', how='inner',
                     suffixes=('_pred', '_actual'))
    
    if len(merged) == 0:
        print("⚠️  No overlapping dates between predictions and actual data")
        return
    
    # Calculate errors
    merged['error'] = merged['predicted_nvidia_revenue_millions'] - merged['data_center_revenue_millions']
    merged['abs_error'] = abs(merged['error'])
    merged['pct_error'] = (merged['abs_error'] / merged['data_center_revenue_millions']) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Predictions vs Actual
    ax1.plot(merged['date'], merged['data_center_revenue_millions'],
            linewidth=2.5, color='#A23B72', marker='o', markersize=5,
            label='Actual Revenue', alpha=0.8)
    ax1.plot(merged['date'], merged['predicted_nvidia_revenue_millions'],
            linewidth=2, color='#F18F01', marker='x', markersize=6,
            label='Predicted Revenue', alpha=0.8, linestyle='--')
    
    # Add error bars
    ax1.errorbar(merged['date'], merged['predicted_nvidia_revenue_millions'],
                yerr=merged['abs_error'], fmt='none', color='red', alpha=0.3,
                capsize=3, label='Error')
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue (Millions USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Accuracy: Actual vs Predicted Revenue', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentage Error
    colors = ['red' if x > 0 else 'green' for x in merged['error']]
    ax2.bar(merged['date'], merged['pct_error'], color=colors, alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error by Date', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    mae = merged['abs_error'].mean()
    mape = merged['pct_error'].mean()
    rmse = np.sqrt((merged['error']**2).mean())
    
    stats_text = f'MAE: ${mae:.2f}M\nMAPE: {mape:.2f}%\nRMSE: ${rmse:.2f}M'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = NOTEBOOKS_DIR / "prediction_accuracy.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_trading_signals(taiwan_df, nvidia_df, predictions_df):
    """Plot trading signals: Taiwan exports with buy signals, overlay Nvidia revenue."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Merge data
    merged = pd.merge(taiwan_df, nvidia_df, on='date', how='inner', suffixes=('_taiwan', '_nvidia'))
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Calculate export changes (for signal detection)
    merged['export_change'] = merged['monthly_exports_usd_millions'].pct_change(periods=1)
    merged['export_ma3'] = merged['monthly_exports_usd_millions'].rolling(3).mean()
    merged['export_ma3_change'] = merged['export_ma3'].pct_change(periods=1)
    
    # Identify buy signals: significant export increases
    threshold = 0.15  # 15% increase
    merged['buy_signal'] = (merged['export_change'] > threshold) | (merged['export_ma3_change'] > threshold * 0.5)
    
    # Plot 1: Taiwan exports with buy signals
    ax1.plot(merged['date'], merged['monthly_exports_usd_millions'], 
             linewidth=2, color='#2E86AB', label='Taiwan Exports', alpha=0.7)
    
    # Mark buy signals
    buy_dates = merged[merged['buy_signal']]['date']
    buy_values = merged[merged['buy_signal']]['monthly_exports_usd_millions']
    ax1.scatter(buy_dates, buy_values, color='green', s=200, marker='^', 
               zorder=5, label='Buy Signal (Export Spike)', edgecolors='darkgreen', linewidths=2)
    
    # Annotate significant signals
    for idx, row in merged[merged['buy_signal']].iterrows():
        if row['export_change'] > 0.2:  # Only annotate major spikes
            ax1.annotate(f"+{row['export_change']*100:.0f}%", 
                        xy=(row['date'], row['monthly_exports_usd_millions']),
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # Highlight surge period
    surge_start = pd.Timestamp('2023-01-01')
    ax1.axvspan(surge_start, merged['date'].max(), alpha=0.15, color='orange', label='Surge Period')
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Taiwan Exports (Millions USD)', fontsize=12, fontweight='bold', color='#2E86AB')
    ax1.set_title('Trading Signals: Taiwan Export Spikes → Nvidia Revenue (1-3 Month Lead)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    
    # Plot 2: Nvidia revenue with forward-shifted export signals
    ax2_twin = ax1.twinx()
    ax2_twin.plot(merged['date'], merged['data_center_revenue_millions'],
                 linewidth=2.5, color='#A23B72', marker='o', markersize=4,
                 label='Nvidia Revenue', alpha=0.8)
    
    # Shift buy signals forward by 2 months (typical lead time)
    shifted_signals = merged[merged['buy_signal']].copy()
    shifted_signals['date'] = shifted_signals['date'] + pd.DateOffset(months=2)
    shifted_signals = shifted_signals[shifted_signals['date'] <= merged['date'].max()]
    
    if len(shifted_signals) > 0:
        ax2_twin.scatter(shifted_signals['date'], 
                        merged[merged['date'].isin(shifted_signals['date'])]['data_center_revenue_millions'],
                        color='purple', s=150, marker='*', zorder=5,
                        label='Expected Revenue Impact (2mo lead)', edgecolors='darkviolet', linewidths=2)
    
    ax2_twin.set_ylabel('Nvidia Revenue (Millions USD)', fontsize=12, fontweight='bold', color='#A23B72')
    ax2_twin.tick_params(axis='y', labelcolor='#A23B72')
    ax2_twin.legend(loc='upper right', fontsize=10)
    
    # Bottom plot: Revenue with predictions
    if predictions_df is not None:
        pred_merged = pd.merge(predictions_df, nvidia_df, on='date', how='left', 
                               suffixes=('_pred', '_actual'))
        ax2.plot(pred_merged['date'], pred_merged['data_center_revenue_millions'],
                linewidth=2.5, color='#A23B72', marker='o', markersize=4,
                label='Nvidia Actual Revenue', alpha=0.8)
        ax2.plot(pred_merged['date'], pred_merged['predicted_nvidia_revenue_millions'],
                linewidth=2, color='#F18F01', marker='x', markersize=6,
                label='Predicted Revenue', alpha=0.8, linestyle='--')
    else:
        recent_merged = merged.tail(24)
        ax2.plot(recent_merged['date'], recent_merged['data_center_revenue_millions'],
                linewidth=2.5, color='#A23B72', marker='o', markersize=4,
                label='Nvidia Revenue', alpha=0.8)
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Revenue (Millions USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Nvidia Revenue: Actual vs Predicted', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = NOTEBOOKS_DIR / "trading_signals.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_earnings_surprise_table(nvidia_df, predictions_df):
    """Create earnings surprise prediction table for next quarters."""
    if predictions_df is None:
        print("   ⚠️  No predictions available for earnings surprise table")
        return
    
    # Get quarterly data (Nvidia reports quarterly)
    quarterly_file = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_quarterly.csv"
    if not quarterly_file.exists():
        print("   ⚠️  Quarterly data not found")
        return
    
    quarterly_df = pd.read_csv(quarterly_file)
    quarterly_df['quarter_end_date'] = pd.to_datetime(quarterly_df['quarter_end_date'])
    
    # Convert monthly predictions to quarterly (sum 3 months)
    predictions_df['year'] = predictions_df['date'].dt.year
    predictions_df['quarter'] = predictions_df['date'].dt.quarter
    
    # Get future quarters (beyond last actual data, or use recent predictions if no future data)
    last_actual_date = quarterly_df['quarter_end_date'].max()
    future_predictions = predictions_df[predictions_df['date'] > last_actual_date].copy()
    
    # If no future predictions, use the most recent predictions (for demo purposes)
    if len(future_predictions) == 0:
        # Use last 12 months of predictions
        future_predictions = predictions_df.tail(12).copy()
        print("   ℹ️  Using recent predictions (no future data available)")
    
    # Aggregate monthly predictions to quarterly
    quarterly_predictions = future_predictions.groupby(['year', 'quarter'])['predicted_nvidia_revenue_millions'].sum().reset_index()
    # Convert year and quarter to integers, then create quarter end dates
    quarterly_predictions['year'] = quarterly_predictions['year'].astype(int)
    quarterly_predictions['quarter'] = quarterly_predictions['quarter'].astype(int)
    quarterly_predictions['quarter_end'] = pd.to_datetime(
        quarterly_predictions.apply(lambda x: f"{int(x['year'])}-{int(x['quarter']*3)}-01", axis=1)
    ) - pd.DateOffset(days=1)
    
    # Create table data
    # For demo: use recent actual quarters as "consensus" estimate (in real scenario, get from Bloomberg/Refinitiv)
    recent_actual = quarterly_df.tail(4).copy()
    avg_growth = recent_actual['quarterly_revenue_millions'].pct_change().mean()
    
    table_data = []
    for _, pred_row in quarterly_predictions.head(4).iterrows():
        # Estimate "consensus" as recent average with slight growth
        if len(recent_actual) > 0:
            base_consensus = recent_actual['quarterly_revenue_millions'].iloc[-1] * (1 + avg_growth)
        else:
            base_consensus = pred_row['predicted_nvidia_revenue_millions'] * 0.95  # Assume 5% below
        
        model_pred = pred_row['predicted_nvidia_revenue_millions']
        surprise = model_pred - base_consensus
        surprise_pct = (surprise / base_consensus) * 100
        
        # Determine confidence (based on how far out the prediction is)
        quarters_ahead = len(quarterly_predictions[quarterly_predictions['quarter_end'] <= pred_row['quarter_end']])
        confidence = max(0.6, 1.0 - (quarters_ahead - 1) * 0.1)  # Decrease confidence for further out
        
        table_data.append({
            'Quarter': f"{pred_row['year']}Q{pred_row['quarter']}",
            'Model Prediction ($M)': f"${model_pred:,.0f}",
            'Consensus Est. ($M)': f"${base_consensus:,.0f}",
            'Surprise ($M)': f"${surprise:,.0f}",
            'Surprise %': f"{surprise_pct:+.1f}%",
            'Confidence': f"{confidence*100:.0f}%"
        })
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=[[row[col] for col in table_data[0].keys()] for row in table_data],
                    colLabels=list(table_data[0].keys()),
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(table_data[0].keys())):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code surprises
    for i, row in enumerate(table_data, 1):
        surprise_pct = float(row['Surprise %'].replace('%', '').replace('+', ''))
        if surprise_pct > 5:
            color = '#90EE90'  # Light green for positive surprise
        elif surprise_pct < -5:
            color = '#FFB6C1'  # Light red for negative surprise
        else:
            color = '#F0F0F0'  # Gray for neutral
        
        for j in range(len(row)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Earnings Surprise Predictions: Model vs Consensus\n(High Confidence = Near-term, Lower = Further Out)', 
             fontsize=14, fontweight='bold', pad=20)
    
    filepath = NOTEBOOKS_DIR / "earnings_surprise_predictions.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_regime_analysis(taiwan_df, nvidia_df, predictions_df):
    """Plot regime analysis: pre-surge vs post-surge performance."""
    # Merge data
    merged = pd.merge(taiwan_df, nvidia_df, on='date', how='inner', suffixes=('_taiwan', '_nvidia'))
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Define regime split
    surge_date = pd.Timestamp('2023-01-01')
    pre_surge = merged[merged['date'] < surge_date].copy()
    post_surge = merged[merged['date'] >= surge_date].copy()
    
    if len(pre_surge) == 0 or len(post_surge) == 0:
        print("   ⚠️  Insufficient data for regime analysis")
        return
    
    # Calculate correlations for each regime
    pre_corr = pre_surge['monthly_exports_usd_millions'].corr(pre_surge['data_center_revenue_millions'])
    post_corr = post_surge['monthly_exports_usd_millions'].corr(post_surge['data_center_revenue_millions'])
    
    # Calculate model performance if predictions exist
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Pre-surge correlation
    ax1.scatter(pre_surge['monthly_exports_usd_millions'], 
               pre_surge['data_center_revenue_millions'],
               alpha=0.6, s=80, color='#2E86AB', edgecolors='black', linewidth=0.5)
    z1 = np.polyfit(pre_surge['monthly_exports_usd_millions'], 
                   pre_surge['data_center_revenue_millions'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(pre_surge['monthly_exports_usd_millions'], 
            p1(pre_surge['monthly_exports_usd_millions']), 
            "r--", alpha=0.8, linewidth=2)
    ax1.set_xlabel('Taiwan Exports (Millions USD)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Nvidia Revenue (Millions USD)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Pre-Surge Regime (Before 2023)\nCorrelation: {pre_corr:.3f}', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'n={len(pre_surge)} months', transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Post-surge correlation
    ax2.scatter(post_surge['monthly_exports_usd_millions'], 
               post_surge['data_center_revenue_millions'],
               alpha=0.6, s=80, color='#F18F01', edgecolors='black', linewidth=0.5)
    z2 = np.polyfit(post_surge['monthly_exports_usd_millions'], 
                   post_surge['data_center_revenue_millions'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(post_surge['monthly_exports_usd_millions'], 
            p2(post_surge['monthly_exports_usd_millions']), 
            "r--", alpha=0.8, linewidth=2)
    ax2.set_xlabel('Taiwan Exports (Millions USD)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Nvidia Revenue (Millions USD)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Post-Surge Regime (2023+)\nCorrelation: {post_corr:.3f}', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f'n={len(post_surge)} months', transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Time series with regime split
    ax3.plot(merged['date'], merged['monthly_exports_usd_millions'], 
            linewidth=2, color='#2E86AB', label='Taiwan Exports', alpha=0.7)
    ax3.axvline(x=surge_date, color='red', linestyle='--', linewidth=2, label='Regime Change (2023)')
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Exports (Millions USD)', fontsize=11, fontweight='bold', color='#2E86AB')
    ax3.set_title('Taiwan Exports: Regime Transition', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='#2E86AB')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(merged['date'], merged['data_center_revenue_millions'],
                 linewidth=2.5, color='#A23B72', marker='o', markersize=3,
                 label='Nvidia Revenue', alpha=0.7)
    ax3_twin.set_ylabel('Revenue (Millions USD)', fontsize=11, fontweight='bold', color='#A23B72')
    ax3_twin.tick_params(axis='y', labelcolor='#A23B72')
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Pre-Surge', 'Post-Surge', 'Change'],
        ['Correlation', f'{pre_corr:.3f}', f'{post_corr:.3f}', f'{post_corr-pre_corr:+.3f}'],
        ['Avg Exports ($M)', f'${pre_surge["monthly_exports_usd_millions"].mean():.0f}', 
         f'${post_surge["monthly_exports_usd_millions"].mean():.0f}',
         f'{((post_surge["monthly_exports_usd_millions"].mean() / pre_surge["monthly_exports_usd_millions"].mean() - 1) * 100):+.0f}%'],
        ['Avg Revenue ($M)', f'${pre_surge["data_center_revenue_millions"].mean():.0f}',
         f'${post_surge["data_center_revenue_millions"].mean():.0f}',
         f'{((post_surge["data_center_revenue_millions"].mean() / pre_surge["data_center_revenue_millions"].mean() - 1) * 100):+.0f}%'],
        ['Months', f'{len(pre_surge)}', f'{len(post_surge)}', '-'],
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code changes
    for i in range(1, len(summary_data)-1):
        change_val = summary_data[i+1][3]
        if '+' in change_val:
            color = '#90EE90'
        elif '-' in change_val and change_val != '-':
            color = '#FFB6C1'
        else:
            color = '#F0F0F0'
        table[(i, 3)].set_facecolor(color)
    
    ax4.set_title('Regime Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Regime Analysis: Pre-Surge vs Post-Surge Performance', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filepath = NOTEBOOKS_DIR / "regime_analysis.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    taiwan_df, nvidia_df, predictions_df = load_data()
    print(f"   Taiwan exports: {len(taiwan_df)} months")
    print(f"   Nvidia revenue: {len(nvidia_df)} months")
    if predictions_df is not None:
        print(f"   Predictions: {len(predictions_df)} months")
    
    # Generate plots
    print("\n2. Generating visualizations...")
    
    print("\n   → Taiwan exports history...")
    plot_taiwan_exports(taiwan_df)
    
    print("\n   → Nvidia actual vs predicted revenue...")
    plot_nvidia_vs_predictions(nvidia_df, predictions_df)
    
    print("\n   → Combined comparison (dual y-axis)...")
    plot_combined_comparison(taiwan_df, nvidia_df, predictions_df)
    
    print("\n   → Correlation scatter plot...")
    correlation = plot_correlation_scatter(taiwan_df, nvidia_df)
    print(f"      Correlation coefficient: {correlation:.3f}")
    
    if predictions_df is not None:
        print("\n   → Prediction accuracy analysis...")
        plot_prediction_accuracy(nvidia_df, predictions_df)
    
    # High-value outputs for hedge fund interview
    print("\n3. Generating high-value outputs for hedge fund analysis...")
    
    print("\n   → Trading signals chart...")
    plot_trading_signals(taiwan_df, nvidia_df, predictions_df)
    
    print("\n   → Earnings surprise predictions...")
    plot_earnings_surprise_table(nvidia_df, predictions_df)
    
    print("\n   → Regime analysis...")
    plot_regime_analysis(taiwan_df, nvidia_df, predictions_df)
    
    print("\n" + "=" * 60)
    print("✓ All visualizations saved to notebooks/ directory")
    print("=" * 60)

if __name__ == "__main__":
    main()
