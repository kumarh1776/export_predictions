"""Process Nvidia quarterly data center revenue data."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def parse_value(value_str):
    value_str = value_str.strip()
    if value_str.upper().endswith('B'):
        return float(value_str[:-1]) * 1000
    elif value_str.upper().endswith('M'):
        return float(value_str[:-1])
    else:
        num = float(value_str)
        return num * 1000 if num > 10 else num

def get_quarter_end_date(date):
    """Convert report date to quarter end date based on Nvidia's fiscal calendar."""
    month = date.month
    
    # Nvidia fiscal quarters end in: Jan (Q4), Apr (Q1), Jul (Q2), Oct (Q3)
    if month in [1, 2, 3]:
        return pd.Timestamp(year=date.year, month=1, day=1)  # Q4 ends in Jan
    elif month in [4, 5, 6]:
        return pd.Timestamp(year=date.year, month=4, day=1)  # Q1 ends in Apr
    elif month in [7, 8, 9]:
        return pd.Timestamp(year=date.year, month=7, day=1)  # Q2 ends in Jul
    else:
        return pd.Timestamp(year=date.year, month=10, day=1)  # Q3 ends in Oct

def process_nvidia_quarterly_data():
    """Process and save Nvidia quarterly revenue data."""
    quarterly_data = [
        ("October 26, 2025", "51.22B"),
        ("July 27, 2025", "41.10B"),
        ("April 27, 2025", "39.11B"),
        ("January 26, 2025", "35.58B"),
        ("October 31, 2024", "30.77B"),
        ("October 27, 2024", "30.77B"),
        ("July 28, 2024", "26.27B"),
        ("April 28, 2024", "22.56B"),
        ("January 28, 2024", "18.40B"),
        ("October 29, 2023", "14.51B"),
        ("July 31, 2023", "10.32B"),
        ("July 30, 2023", "10.32B"),
        ("April 30, 2023", "4.284B"),
        ("January 29, 2023", "3.616B"),
        ("October 30, 2022", "3.833B"),
        ("July 31, 2022", "3.806B"),
        ("May 01, 2022", "3.75B"),
        ("January 30, 2022", "3.263B"),
        ("October 31, 2021", "2.936B"),
        ("August 01, 2021", "2.366B"),
        ("May 02, 2021", "2.048B"),
        ("January 31, 2021", "1.903B"),
        ("October 25, 2020", "1.90B"),
        ("July 26, 2020", "1.752B"),
        ("April 26, 2020", "1.141B"),
        ("January 26, 2020", "968.00M"),
    ]
    
    parsed_data = []
    seen_quarters = set()
    
    for date_str, value_str in quarterly_data:
        date = pd.to_datetime(date_str)
        quarter_end = get_quarter_end_date(date)
        quarter_key = (quarter_end.year, quarter_end.month)
        
        if quarter_key in seen_quarters:
            continue
        seen_quarters.add(quarter_key)
        
        parsed_data.append({
            'quarter_end_date': quarter_end,
            'quarterly_revenue_millions': parse_value(value_str)
        })
    
    df_quarterly = pd.DataFrame(parsed_data)
    df_quarterly = df_quarterly.sort_values('quarter_end_date').reset_index(drop=True)
    
    # Add fiscal quarter labels
    df_quarterly['fiscal_year'] = df_quarterly['quarter_end_date'].apply(
        lambda x: x.year if x.month != 1 else x.year - 1
    )
    df_quarterly['fiscal_quarter'] = df_quarterly['quarter_end_date'].apply(
        lambda x: {1: 4, 4: 1, 7: 2, 10: 3}[x.month]
    )
    
    print(f"Processing {len(df_quarterly)} quarters")
    print(f"Date range: {df_quarterly['quarter_end_date'].min()} to {df_quarterly['quarter_end_date'].max()}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_quarterly.csv"
    df_quarterly.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return df_quarterly

if __name__ == "__main__":
    process_nvidia_quarterly_data()
