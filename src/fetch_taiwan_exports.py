"""
Fetch Taiwan exports data from Taiwan Customs Statistics Database.

Website: https://portal.sw.nat.gov.tw/APGA/GA30E
Commodity codes: 8471100000,8471200000,8471410000,8471490000,8471800000
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

BASE_URL = "https://portal.sw.nat.gov.tw/APGA/GA30E"
COMMODITY_CODES = "8471100000,8471200000,8471410000,8471490000,8471800000"

def fetch_with_selenium():
    """Fetch data using Selenium (handles CAPTCHA)."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select, WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        import time
    except ImportError:
        print("❌ Selenium not installed. Install with: pip install selenium")
        print("   Also install ChromeDriver: https://chromedriver.chromium.org/")
        return None
    
    print("=" * 60)
    print("Fetching Taiwan Exports Data")
    print("=" * 60)
    
    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Set download directory
    download_dir = str(RAW_DATA_DIR.absolute())
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    driver = None
    try:
        print("\n1. Opening browser...")
        driver = webdriver.Chrome(options=chrome_options)
        
        print("2. Accessing Taiwan Customs Statistics Database...")
        driver.get(BASE_URL)
        
        # Wait for page
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "form"))
        )
        
        print("3. Filling form...")
        
        # Try to fill form fields (selectors may need adjustment)
        try:
            # Select Exports
            exports_radios = driver.find_elements(By.XPATH, "//input[contains(@value, 'Export') or contains(@value, '出口')]")
            if exports_radios:
                exports_radios[0].click()
            
            # Select Monthly
            try:
                monthly = Select(driver.find_element(By.NAME, "periodicity"))
                monthly.select_by_visible_text("Monthly")
            except:
                pass
            
            # Enter commodity codes
            commodity_inputs = driver.find_elements(By.XPATH, "//input[contains(@name, 'commodity') or contains(@id, 'commodity')]")
            if commodity_inputs:
                commodity_inputs[0].clear()
                commodity_inputs[0].send_keys(COMMODITY_CODES)
            
            # Select World
            try:
                world = Select(driver.find_element(By.NAME, "partner"))
                world.select_by_visible_text("World")
            except:
                pass
            
            # Select USD
            try:
                usd = Select(driver.find_element(By.NAME, "measure"))
                usd.select_by_visible_text("USD")
            except:
                pass
            
            # Select CSV export
            csv_radios = driver.find_elements(By.XPATH, "//input[contains(@value, 'csv') or contains(@value, 'CSV')]")
            if csv_radios:
                csv_radios[0].click()
            
            print("   ✓ Form filled")
            
        except Exception as e:
            print(f"   ⚠️  Could not auto-fill all fields: {e}")
            print("   Please fill the form manually:")
            print(f"   - Commodity codes: {COMMODITY_CODES}")
            print("   - Total Exports, Monthly, World, USD, CSV")
        
        print("\n4. Waiting for CAPTCHA...")
        print("   ⚠️  Please solve the CAPTCHA in the browser window")
        input("   Press Enter after solving CAPTCHA and submitting the form...")
        
        # Wait for download
        print("\n5. Waiting for download...")
        time.sleep(5)
        
        # Check for downloaded file
        downloaded_files = list(RAW_DATA_DIR.glob("*.csv"))
        if downloaded_files:
            latest_file = max(downloaded_files, key=lambda p: p.stat().st_mtime)
            print(f"   ✓ Downloaded: {latest_file.name}")
            
            # Rename to standard name
            target_file = RAW_DATA_DIR / "taiwan_exports_history.csv"
            if latest_file != target_file:
                latest_file.rename(target_file)
                print(f"   ✓ Renamed to: taiwan_exports_history.csv")
            
            return target_file
        else:
            print("   ⚠️  No CSV file found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        if driver:
            print("\n6. Closing browser...")
            time.sleep(2)
            driver.quit()

def main():
    """Main function."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_file = fetch_with_selenium()
    
    if csv_file:
        print(f"\n✓ Success! File saved to: {csv_file}")
        print("\nNext step: Run 'python src/clean_data.py' to process the data")
    else:
        print("\n⚠️  Fetch failed. Options:")
        print("1. Try again")
        print("2. Manually download CSV from the website")
        print("3. Save to: data/raw/taiwan_exports_history.csv")

if __name__ == "__main__":
    main()
