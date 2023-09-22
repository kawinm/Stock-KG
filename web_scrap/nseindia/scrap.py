import time 
 
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver import Chrome 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
#from selenium_stealth import stealth
#from fake_useragent import UserAgent

#import undetected_chromedriver as uc

import pickle

from bs4 import BeautifulSoup
import csv

extracted_data = {}
with open("nifty500_constituents.csv") as ticker_file:
    ticker_reader = csv.DictReader(ticker_file)

    for row in ticker_reader:
        ticker = row["Symbol"]
        print(ticker)
        retrieve_url = "https://www.nseindia.com/get-quotes/equity?symbol="+ticker
        my_options = webdriver.ChromeOptions()
        #my_options.add_argument(f'user-data-dir=Scraper')
        my_options.add_argument('--enable-javascript')
        #my_options.add_argument("--headless")
        my_options.add_argument('--disable-blink-features=AutomationControlled')
        my_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        my_options.add_experimental_option('useAutomationExtension', False)
        #my_options.add_argument(f'user-agent={userAgent}')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=my_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")

        #set_viewport_size(driver, 800, 600)

        driver.get(retrieve_url) 
        time.sleep(4)

        links = []
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.findAll('td', recursive=True)

        data = []
        for n, td in enumerate(table):
            if n in [65, 78, 80, 81, 82, 83]:
                data.append(td.text.strip())
                print(f"index {n}: {td.text.strip()}")
        extracted_data[ticker] = data

with open("sector_nse.pkl", "wb") as f:
    pickle.dump(extracted_data, f)

