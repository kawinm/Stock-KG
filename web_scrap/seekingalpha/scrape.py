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


# pass the defined options and service objects to initialize the web driver 
# stock- mhcp, mdlz
#ticker_list_all = ['CPRT', 'ROST', 'BIDU', 'ODFL', 'KDP', 'NFLX', 'SBUX', 'MELI', 'MRVL', 'BIIB', 'INTC', 'SNPS', 'SGEN', 'DLTR', 'INTU', 'AMZN', 'NTES', 'DDOG', 'CDNS', 'GILD', 'SIRI', 'DOCU', 'ORLY', 'AMGN', 'META', 'FISV', 'ADI', 'MU', 'BKNG', 'SWKS', 'CRWD', 'CEG', 'PCAR', 'ADP', 'AMAT', 'GOOG', 'CTAS', 'TXN', 'NVDA', 'ZM', 'ALGN', 'WDAY', 'NXPI', 'LRCX', 'TSLA', 'ASML', 'CMCSA', 'ENPH', 'PANW', 'XEL', 'ANSS', 'MTCH', 'TEAM', 'AVGO', 'MRNA', 'VRSN', 'KLAC', 'LULU', 'VRTX', 'GOOGL', 'PAYX', 'ADBE', 'ILMN', 'AAPL', 'AEP', 'CSCO', 'LCID', 'VRSK', 'AMD', 'EA', 'COST', 'EBAY', 'ISRG', 'ABNB', 'PEP', 'FAST', 'MSFT', 'ATVI', 'REGN', 'PDD', 'CSX', 'MNST', 'IDXX', 'SPLK', 'KHC', 'JD', 'QCOM', 'ZS', 'ADSK', 'WBA', 'CTSH', 'FTNT', 'DXCM', 'PYPL', 'HON', 'CHTR', 'MAR', 'TMUS', 'EXC', 'AZN']

ticker_list = ['CPRT', 'ROST', 'BIDU', 'ODFL', 'KDP', 'NFLX', 'SBUX', 'MELI', 'MRVL', 'BIIB', 'INTC', 'SNPS', 'SGEN', 'DLTR', 'INTU', 'AMZN', 'NTES', 'CDNS', 'GILD', 'MDLZ', 'MCHP', 'SIRI', 'ORLY', 'AMGN', 'META', 'FISV', 'ADI', 'MU', 'BKNG', 'SWKS', 'PCAR', 'ADP', 'AMAT', 'GOOG', 'CTAS', 'TXN', 'NVDA', 'ALGN', 'NXPI', 'LRCX', 'TSLA', 'ASML', 'CMCSA', 'ENPH', 'PANW', 'XEL', 'ANSS', 'MTCH', 'AVGO', 'VRSN', 'KLAC', 'LULU', 'VRTX', 'GOOGL', 'PAYX', 'ADBE', 'ILMN', 'AAPL', 'AEP', 'CSCO', 'VRSK', 'AMD', 'EA', 'COST', 'EBAY', 'ISRG', 'PEP', 'FAST', 'MSFT', 'ATVI', 'REGN', 'CSX', 'MNST', 'IDXX', 'SPLK', 'QCOM', 'ADSK', 'WBA', 'CTSH', 'FTNT', 'DXCM', 'HON', 'CHTR', 'MAR', 'TMUS', 'EXC', 'AZN']
news_type = "earning"   # mna, dividend, all

ii = 0
for page in range(1, 2):

    print(page)
    
    time.sleep(5)
    for ticker in ticker_list[20:]:

        print(ticker)

        filename = "scrap/sa_" + news_type + "_"+ticker+"_p"+str(page)+".html"

        if os.path.isfile(filename):
            print("file exists")
            continue

        my_options = webdriver.ChromeOptions()
        #my_options.add_argument(f'user-data-dir=Scraper')
        my_options.add_argument('--enable-javascript')
        my_options.add_argument('--disable-blink-features=AutomationControlled')
        my_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        my_options.add_experimental_option('useAutomationExtension', False)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=my_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")

        if news_type == "dividend":
            url = 'https://seekingalpha.com/symbol/'+ticker+'/news?filter=dividend_news&page='+str(page)
        elif news_type == "mna":
            url = 'https://seekingalpha.com/symbol/'+ticker+'/news?filter=m_n_a_news&page='+str(page)
        elif news_type == "earning":
            url = 'https://seekingalpha.com/symbol/'+ticker+'/news?filter=earnings_news&page='+str(page)

        driver.get(url) 
        time.sleep(10)

        #print(driver.page_source)
        #try:
        #    content = driver.find_element(By.CSS_SELECTOR, "div[class*='iX-EL'")
        #except:
        #    break

        w = open(filename, "w")
        w.write(driver.page_source)
        w.close()

        driver.quit()

#p2 - INTC, MDLZ, AMGN, ADI, MAR, EXC
#prob- MCHP

# mna - 
"""
odfl1, kdp1, nflx2, sbux1, meli1, mrvl2, biib1
'intc': 4, 'snps': 1, 'sgen': 1, 'dltr': 1, 'intu': 1, 
'amzn': 5, 'ntes': 1, 'ddog': 1, 'cdns': 1, 'gild': 2,
'mdlz': 2, 'siri': 2, 'docu': 1, 'amgn': 2, 'meta':  ,
'adi': 2, 'mu': 2, 'goog': 5, 'nvda': 2, 'nxpi': 2,
'lrcx': 1, 'tsla': 3, 'cmcsa': 5, 
"""