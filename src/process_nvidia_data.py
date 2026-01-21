"""
Process Nvidia quarterly data center revenue and convert to monthly data.
Each quarter's revenue is evenly divided across its 3 months.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def parse_value(value_str):
    """
    Parse value string (e.g., '51.22B', '968.00M') to float in millions.
    
    Args:
        value_str: String like '51.22B' or '968.00M'
    
    Returns:
        Float value in millions
    """
    # Remove any whitespace
    value_str = value_str.strip()
    
    # Check if it ends with B (billions) or M (millions)
    if value_str.upper().endswith('B'):
        # Convert billions to millions
        num_str = value_str[:-1]
        return float(num_str) * 1000
    elif value_str.upper().endswith('M'):
        # Already in millions
        num_str = value_str[:-1]
        return float(num_str)
    else:
        # Try to parse as number (assume billions if large)
        num = float(value_str)
        if num > 10:  # Likely billions
            return num * 1000
        return num

def get_quarter_months(date):
    """
    Get the 3 months that belong to the quarter ending on this date.
    Nvidia's fiscal quarters:
    - Q1: Feb, Mar, Apr (ends in April)
    - Q2: May, Jun, Jul (ends in July)
    - Q3: Aug, Sep, Oct (ends in October)
    - Q4: Nov, Dec, Jan (ends in January)
    
    Args:
        date: pandas Timestamp
    
    Returns:
        List of 3 pandas Timestamps for the months in that quarter
    """
    month = date.month
    
    if month == 1:  # January - Q4 (Nov, Dec, Jan)
        months = [11, 12, 1]
        year_offset = [0, 0, 0]  # Nov and Dec are previous year
    elif month == 4:  # April - Q1 (Feb, Mar, Apr)
        months = [2, 3, 4]
        year_offset = [0, 0, 0]
    elif month == 5:  # May - Q1 (Feb, Mar, Apr) - sometimes reported in May
        months = [2, 3, 4]
        year_offset = [0, 0, 0]
    elif month == 7:  # July - Q2 (May, Jun, Jul)
        months = [5, 6, 7]
        year_offset = [0, 0, 0]
    elif month == 8:  # August - Q2 (May, Jun, Jul) - sometimes reported in August
        months = [5, 6, 7]
        year_offset = [0, 0, 0]
    elif month == 10:  # October - Q3 (Aug, Sep, Oct)
        months = [8, 9, 10]
        year_offset = [0, 0, 0]
    else:
        # Default: assume it's the end of a quarter
        if month <= 3:
            months = [11, 12, 1]
            year_offset = [0, 0, 0]
        elif month <= 6:
            months = [2, 3, 4]
            year_offset = [0, 0, 0]
        elif month <= 9:
            months = [5, 6, 7]
            year_offset = [0, 0, 0]
        else:
            months = [8, 9, 10]
            year_offset = [0, 0, 0]
    
    # Create dates for each month
    quarter_dates = []
    for m, offset in zip(months, year_offset):
        if m == 11 or m == 12:  # November/December are previous calendar year
            year = date.year - 1
        else:
            year = date.year
        quarter_dates.append(pd.Timestamp(year=year, month=m, day=1))
    
    return quarter_dates

def process_nvidia_quarterly_data():
    """
    Process quarterly Nvidia data and convert to monthly.
    """
    # Raw quarterly data
    quarterly_data = [
        ("October 26, 2025", "51.22B"),
        ("July 27, 2025", "41.10B"),
        ("April 27, 2025", "39.11B"),
        ("January 26, 2025", "35.58B"),
        ("October 31, 2024", "30.77B"),
        ("October 27, 2024", "30.77B"),  # Duplicate
        ("July 28, 2024", "26.27B"),
        ("April 28, 2024", "22.56B"),
        ("January 28, 2024", "18.40B"),
        ("October 29, 2023", "14.51B"),
        ("July 31, 2023", "10.32B"),
        ("July 30, 2023", "10.32B"),  # Duplicate
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
    
    # Parse dates and values
    parsed_data = []
    seen_quarters = set()  # Track duplicates
    
    for date_str, value_str in quarterly_data:
        # Parse date
        date = pd.to_datetime(date_str)
        
        # Create quarter key (year-month) to detect duplicates
        quarter_key = (date.year, date.month)
        if quarter_key in seen_quarters:
            print(f"Skipping duplicate: {date_str}")
            continue
        seen_quarters.add(quarter_key)
        
        # Parse value to millions
        value_millions = parse_value(value_str)
        
        parsed_data.append({
            'quarter_end_date': date,
            'quarterly_revenue_millions': value_millions
        })
    
    # Convert to DataFrame
    df_quarterly = pd.DataFrame(parsed_data)
    df_quarterly = df_quarterly.sort_values('quarter_end_date').reset_index(drop=True)
    
    print("=" * 60)
    print("Nvidia Quarterly Data Center Revenue")
    print("=" * 60)
    print(f"\nTotal quarters: {len(df_quarterly)}")
    print(f"Date range: {df_quarterly['quarter_end_date'].min()} to {df_quarterly['quarter_end_date'].max()}")
    print("\nQuarterly Data:")
    print(df_quarterly.to_string(index=False))
    
    # Convert to monthly
    monthly_data = []
    
    for _, row in df_quarterly.iterrows():
        quarter_end = row['quarter_end_date']
        quarterly_revenue = row['quarterly_revenue_millions']
        monthly_revenue = quarterly_revenue / 3.0
        
        # Get the 3 months for this quarter
        quarter_months = get_quarter_months(quarter_end)
        
        for month_date in quarter_months:
            monthly_data.append({
                'date': month_date,
                'data_center_revenue_millions': monthly_revenue,
                'quarter_end_date': quarter_end,
                'quarterly_revenue_millions': quarterly_revenue
            })
    
    # Convert to DataFrame
    df_monthly = pd.DataFrame(monthly_data)
    df_monthly = df_monthly.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates (in case quarters overlap)
    df_monthly = df_monthly.drop_duplicates(subset=['date'], keep='first')
    df_monthly = df_monthly.sort_values('date').reset_index(drop=True)
    
    # Select final columns
    df_final = df_monthly[['date', 'data_center_revenue_millions']].copy()
    
    print("\n" + "=" * 60)
    print("Monthly Data (Quarterly Revenue / 3)")
    print("=" * 60)
    print(f"\nTotal months: {len(df_final)}")
    print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print("\nSample monthly data:")
    print(df_final.head(12).to_string(index=False))
    print("\n...")
    print(df_final.tail(12).to_string(index=False))
    
    # Save to CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_monthly.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✓ Monthly data saved to: {output_file}")
    
    # Also save quarterly data for reference
    quarterly_output = PROCESSED_DATA_DIR / "nvidia_data_center_revenue_quarterly.csv"
    df_quarterly.to_csv(quarterly_output, index=False)
    print(f"✓ Quarterly data saved to: {quarterly_output}")
    
    return df_final

if __name__ == "__main__":
    process_nvidia_quarterly_data()
