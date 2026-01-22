"""Clean Taiwan exports data by aggregating commodity codes per month."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def clean_taiwan_exports():
    input_file = RAW_DATA_DIR / "taiwan_exports_history.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_file}")
    
    df = pd.read_csv(input_file)
    
    date_col = None
    value_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['time', 'period', 'date', '年月']):
            date_col = col
        if any(kw in col_lower for kw in ['value', 'usd', '金額', '美金']):
            value_col = col
    
    if date_col is None:
        date_col = 'Time' if 'Time' in df.columns else df.columns[0]
    if value_col is None:
        value_col = 'Value(USD$ 1000)' if 'Value(USD$ 1000)' in df.columns else df.columns[-1]
    
    df['date'] = pd.to_datetime(df[date_col], errors='coerce', format='%Y/%m')
    if df['date'].isna().all():
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    if df[value_col].dtype == 'object':
        df[value_col] = df[value_col].astype(str).str.replace(',', '').str.replace('"', '')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    monthly_totals = df.groupby('date')[value_col].sum().reset_index()
    monthly_totals.columns = ['date', 'monthly_exports_usd_thousands']
    monthly_totals['monthly_exports_usd_millions'] = monthly_totals['monthly_exports_usd_thousands'] / 1000
    monthly_totals = monthly_totals.sort_values('date').reset_index(drop=True)
    monthly_totals['year'] = monthly_totals['date'].dt.year
    monthly_totals['month'] = monthly_totals['date'].dt.month
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "taiwan_exports_monthly.csv"
    monthly_totals.to_csv(output_file, index=False)
    
    print(f"Cleaned {len(monthly_totals)} months of data")
    print(f"Date range: {monthly_totals['date'].min()} to {monthly_totals['date'].max()}")
    print(f"Saved to: {output_file}")
    
    return monthly_totals

if __name__ == "__main__":
    clean_taiwan_exports()
