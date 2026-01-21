"""
Data cleaning script for Taiwan exports data.
Aggregates commodity codes per month to get monthly totals.

Can process data from:
1. Manual CSV download from Taiwan Customs Statistics Database
2. Automated fetch via fetch_taiwan_exports.py
"""

import pandas as pd
import os
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def clean_taiwan_exports():
    """
    Clean Taiwan exports CSV by:
    1. Reading the raw CSV
    2. Parsing the Time column to proper datetime
    3. Summing the 3 commodity codes per month
    4. Saving cleaned data
    """
    # Read raw data
    input_file = RAW_DATA_DIR / "taiwan_exports_history.csv"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_file}")
    
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Display basic info
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Try to identify date and value columns (handle different CSV formats)
    date_col = None
    value_col = None
    
    # Check for common column names
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['time', 'period', 'date', '年月']):
            date_col = col
        if any(keyword in col_lower for keyword in ['value', 'usd', '金額', '美金']):
            value_col = col
    
    # Fallback to expected column names
    if date_col is None:
        if 'Time' in df.columns:
            date_col = 'Time'
        elif 'Period' in df.columns:
            date_col = 'Period'
        else:
            date_col = df.columns[0]  # Use first column
    
    if value_col is None:
        if 'Value(USD$ 1000)' in df.columns:
            value_col = 'Value(USD$ 1000)'
        elif 'Value' in df.columns:
            value_col = 'Value'
        else:
            value_col = df.columns[-1]  # Use last column
    
    print(f"Using date column: {date_col}")
    print(f"Using value column: {value_col}")
    
    if date_col in df.columns:
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    if 'Commodity Code' in df.columns:
        print(f"Unique commodity codes: {df['Commodity Code'].nunique()}")
    
    # Convert date column to datetime
    # Try different formats
    df['date'] = pd.to_datetime(df[date_col], errors='coerce', format='%Y/%m')
    if df['date'].isna().all():
        # Try other formats
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Ensure Value column is numeric (remove any commas, handle edge cases)
    if df[value_col].dtype == 'object':
        df[value_col] = df[value_col].astype(str).str.replace(',', '').str.replace('"', '')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Group by date and sum the values (aggregating all commodity codes)
    monthly_totals = df.groupby('date')[value_col].sum().reset_index()
    monthly_totals.columns = ['date', 'monthly_exports_usd_thousands']
    
    # Convert to millions for easier reading (optional)
    monthly_totals['monthly_exports_usd_millions'] = monthly_totals['monthly_exports_usd_thousands'] / 1000
    
    # Sort by date
    monthly_totals = monthly_totals.sort_values('date').reset_index(drop=True)
    
    # Add year and month columns for easier filtering
    monthly_totals['year'] = monthly_totals['date'].dt.year
    monthly_totals['month'] = monthly_totals['date'].dt.month
    
    # Save cleaned data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "taiwan_exports_monthly.csv"
    monthly_totals.to_csv(output_file, index=False)
    
    print(f"\nCleaned data shape: {monthly_totals.shape}")
    print(f"Date range: {monthly_totals['date'].min()} to {monthly_totals['date'].max()}")
    print(f"Total months: {len(monthly_totals)}")
    print(f"\nSample of cleaned data:")
    print(monthly_totals.head(10))
    print(f"\nCleaned data saved to: {output_file}")
    
    return monthly_totals

if __name__ == "__main__":
    clean_taiwan_exports()
