"""Fetch Taiwan exports data from Taiwan Customs Statistics Database."""

import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
BASE_URL = "https://portal.sw.nat.gov.tw/APGA/GA30E"
# Commodity codes for data processing equipment exports
COMMODITY_CODES = "8471100000,8471200000,8471410000,8471490000,8471800000"

def fetch_with_selenium():
    """Automate browser to fetch Taiwan exports data using Selenium."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select, WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        print("Selenium not installed. Install with: pip install selenium")
        return None
    
    print("Fetching Taiwan Exports Data")
    
    # Configure Chrome options for headless/automated browsing
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Set download directory to project's raw data folder
    download_dir = str(RAW_DATA_DIR.absolute())
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(BASE_URL)
        
        # Wait for form to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "form"))
        )
        
        # Auto-fill form fields (may need adjustment if website structure changes)
        try:
            # Select "Exports" option
            exports_radios = driver.find_elements(By.XPATH, "//input[contains(@value, 'Export') or contains(@value, '出口')]")
            if exports_radios:
                exports_radios[0].click()
            
            # Select "Monthly" periodicity
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
            
            # Select "World" as partner country
            try:
                world = Select(driver.find_element(By.NAME, "partner"))
                world.select_by_visible_text("World")
            except:
                pass
            
            # Select "USD" as currency
            try:
                usd = Select(driver.find_element(By.NAME, "measure"))
                usd.select_by_visible_text("USD")
            except:
                pass
            
            # Select CSV export format
            csv_radios = driver.find_elements(By.XPATH, "//input[contains(@value, 'csv') or contains(@value, 'CSV')]")
            if csv_radios:
                csv_radios[0].click()
        except Exception as e:
            print(f"Could not auto-fill all fields: {e}")
            print(f"Please fill manually: Commodity codes: {COMMODITY_CODES}")
        
        # User must solve CAPTCHA manually
        print("Waiting for CAPTCHA...")
        input("Press Enter after solving CAPTCHA and submitting the form...")
        
        # Wait for download to complete
        time.sleep(5)
        
        # Find and rename downloaded CSV file
        downloaded_files = list(RAW_DATA_DIR.glob("*.csv"))
        if downloaded_files:
            latest_file = max(downloaded_files, key=lambda p: p.stat().st_mtime)
            target_file = RAW_DATA_DIR / "taiwan_exports_history.csv"
            if latest_file != target_file:
                latest_file.rename(target_file)
            return target_file
        return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        # Always close browser
        if driver:
            time.sleep(2)
            driver.quit()

def main():
    """Main entry point: fetch data and provide next steps."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = fetch_with_selenium()
    
    if csv_file:
        print(f"Success! File saved to: {csv_file}")
        print("Next step: Run 'python src/clean_data.py'")
    else:
        print("Fetch failed. Manually download CSV and save to: data/raw/taiwan_exports_history.csv")

if __name__ == "__main__":
    main()
