"""Process Nvidia quarterly data center revenue and convert to monthly data."""

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

def get_quarter_months(date):
    month = date.month
    
    if month == 1:
        months = [11, 12, 1]
    elif month in [4, 5]:
        months = [2, 3, 4]
    elif month in [7, 8]:
        months = [5, 6, 7]
    elif month == 10:
        months = [8, 9, 10]
    else:
        if month <= 3:
            months = [11, 12, 1]
        elif month <= 6:
            months = [2, 3, 4]
        elif month <= 9:
            months = [5, 6, 7]
        else:
            months = [8, 9, 10]
    
    quarter_dates = []
    for m in months:
        if m in [11, 12]:
            year = date.year - 1
        else:
            year = date.year
        quarter_dates.append(pd.Timestamp(year=year, month=m, day=1))
    
    return quarter_dates

def process_nvidia_quarterly_data():
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
        quarter_key = (date.year, date.month)
        
        if quarter_key in seen_quarters:
            continue
        seen_quarters.add(quarter_key)
        
        parsed_data.append({
            'quarter_end_date': date,
            'quarterly_revenue_millions': parse_value(value_str)
        })
    
    df_quarterly = pd.DataFrame(parsed_data)
    df_quarterly = df_quarterly.sort_values('quarter_end_date').reset_index(drop=True)
    
    print(f"Processing {len(df_quarterly)} quarters")
    print(f"Date range: {df_quarterly['quarter_end_date'].min()} to {df_quarterly['quarter_end_date'].max()}")
    
    monthly_data = []
    for _, row in df_quarterly.iterrows():
        monthly_revenue = row['quarterly_revenue_millions'] / 3.0
        quarter_months = get_quarter_months(row['quarter_end_date'])
        
        for month_date in quarter_months:
            monthly_data.append({
                'date': month_date,
                'data_center_revenue_millions': monthly_revenue
            })
    
    df_monthly = pd.DataFrame(monthly_data)
    df_monthly = df_monthly.sort_values('date').reset_index(drop=True)
    df_monthly = df_monthly.drop_duplicates(subset=['date'], keep='first')
    
    df_final = df_monthly[['date', 'data_center_revenue_millions']].copy()
    
    print(f"Generated {len(df_final)} monthly records")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_monthly.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    quarterly_output = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_quarterly.csv"
    df_quarterly.to_csv(quarterly_output, index=False)
    
    return df_final

if __name__ == "__main__":
    process_nvidia_quarterly_data()
