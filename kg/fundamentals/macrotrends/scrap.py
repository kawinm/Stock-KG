import time 
 
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver import Chrome 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
from fake_useragent import UserAgent

import undetected_chromedriver as uc

import os

def set_viewport_size(driver, width, height):
    window_size = driver.execute_script("""
        return [window.outerWidth - window.innerWidth + arguments[0],
          window.outerHeight - window.innerHeight + arguments[1]];
        """, width, height)
    driver.set_window_size(*window_size)


# pass the defined options and service objects to initialize the web driver 
# stock- mhcp, mdlz
#ticker_list_all = ['CPRT', 'ROST', 'BIDU', 'ODFL', 'KDP', 'NFLX', 'SBUX', 'MELI', 'MRVL', 'BIIB', 'INTC', 'SNPS', 'SGEN', 'DLTR', 'INTU', 'AMZN', 'NTES', 'DDOG', 'CDNS', 'GILD', 'SIRI', 'DOCU', 'ORLY', 'AMGN', 'META', 'FISV', 'ADI', 'MU', 'BKNG', 'SWKS', 'CRWD', 'CEG', 'PCAR', 'ADP', 'AMAT', 'GOOG', 'CTAS', 'TXN', 'NVDA', 'ZM', 'ALGN', 'WDAY', 'NXPI', 'LRCX', 'TSLA', 'ASML', 'CMCSA', 'ENPH', 'PANW', 'XEL', 'ANSS', 'MTCH', 'TEAM', 'AVGO', 'MRNA', 'VRSN', 'KLAC', 'LULU', 'VRTX', 'GOOGL', 'PAYX', 'ADBE', 'ILMN', 'AAPL', 'AEP', 'CSCO', 'LCID', 'VRSK', 'AMD', 'EA', 'COST', 'EBAY', 'ISRG', 'ABNB', 'PEP', 'FAST', 'MSFT', 'ATVI', 'REGN', 'PDD', 'CSX', 'MNST', 'IDXX', 'SPLK', 'KHC', 'JD', 'QCOM', 'ZS', 'ADSK', 'WBA', 'CTSH', 'FTNT', 'DXCM', 'PYPL', 'HON', 'CHTR', 'MAR', 'TMUS', 'EXC', 'AZN']

ticker_list = ['CPRT', 'ROST', 'BIDU', 'ODFL', 'KDP', 'NFLX', 'SBUX', 'MELI', 'MRVL', 'BIIB', 'INTC', 'SNPS', 'SGEN', 'DLTR', 'INTU', 'AMZN', 'NTES', 'CDNS', 'GILD', 'MDLZ', 'MCHP', 'SIRI', 'ORLY', 'AMGN', 'META', 'FISV', 'ADI', 'MU', 'BKNG', 'SWKS', 'PCAR', 'ADP', 'AMAT', 'GOOG', 'CTAS', 'TXN', 'NVDA', 'ALGN', 'NXPI', 'LRCX', 'TSLA', 'ASML', 'CMCSA', 'ENPH', 'PANW', 'XEL', 'ANSS', 'MTCH', 'AVGO', 'VRSN', 'KLAC', 'LULU', 'VRTX', 'GOOGL', 'PAYX', 'ADBE', 'ILMN', 'AAPL', 'AEP', 'CSCO', 'VRSK', 'AMD', 'EA', 'COST', 'EBAY', 'ISRG', 'PEP', 'FAST', 'MSFT', 'ATVI', 'REGN', 'CSX', 'MNST', 'IDXX', 'SPLK', 'QCOM', 'ADSK', 'WBA', 'CTSH', 'FTNT', 'DXCM', 'HON', 'CHTR', 'MAR', 'TMUS', 'EXC', 'AZN']
event_type = ['pe-ratio', 'peg-ratio', 'price-to-book-value', 'price-to-free-cash-value', 'ps-ratio', 'price-cagr', 'dividends', 'dividends-cagr', 'dividend-yield', 'dividend-yield-on-cost', 'dividend-payout-ratio', 'dividend-payout-ratio-averages', 'current-ratio', 'current-ratio-averages', 'debt-to-equity-ratio', 'debt-to-equity-ratio-averages', 'eps-basic', 'eps-basic-cagr', 'eps-diluted', 'eps-diluted-cagr', 'ev-to-ebit', 'ev-to-ebit-averages', 'ev-to-ebitda', 'ev-to-ebitda-averages', 'ev-to-assets', 'ev-to-assets-averages', 'ev-to-sales', 'ev-to-sales-averages', 'free-cash-flow', 'free-cash-flow-cagr', 'net-income-ttm', 'net-income-ttm-cagr', 'profit-margin', 'profit-margin-averages', 'quick-ratio', 'quick-ratio-averages', 'roa', 'roa-averages', 'roe', 'roe-averages', 'revenue-ttm', 'revenue-ttm-cagr', 'enterprise-value']

print(len(ticker_list)*len(event_type))
ii = 0
for event in event_type:

    print(event)
    
    time.sleep(5)
    for ticker in ticker_list:

        print(ticker)

        filename = "raw_data/" + event + "_"+ ticker + ".html"

        if os.path.isfile(filename):
            print("file exists")
            continue
        
        ua = UserAgent()
        userAgent = ua.random
        print(userAgent)
        
        my_options = webdriver.ChromeOptions()
        #my_options.headless=True
        #my_options.add_argument(f'user-data-dir=Scraper')
        #my_options.add_argument('--enable-javascript')
        my_options.add_argument('--disable-blink-features=AutomationControlled')
        my_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        my_options.add_experimental_option('useAutomationExtension', False)
        my_options.add_argument(f'user-agent={userAgent}')

        driver = Chrome(service=Service(ChromeDriverManager().install()), options=my_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")

        set_viewport_size(driver, 800, 600)
        url = 'https://www.financecharts.com/stocks/' + ticker + '/value/' + event
      

        driver.get(url) 
        time.sleep(40)

        #print(driver.page_source)
        #try:
        #    content = driver.find_element(By.CSS_SELECTOR, "div[class*='iX-EL'")
        #except:
        #    break

        w = open(filename, "w", encoding="utf-8")
        w.write(driver.page_source)
        w.close()

        driver.quit()
