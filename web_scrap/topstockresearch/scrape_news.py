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

from bs4 import BeautifulSoup

import os

def set_viewport_size(driver, width, height):
    window_size = driver.execute_script("""
        return [window.outerWidth - window.innerWidth + arguments[0],
          window.outerHeight - window.innerHeight + arguments[1]];
        """, width, height)
    driver.set_window_size(*window_size)

ticker_list = {}

ticker_file = open('labelmap.txt', "r").readlines()
for line in ticker_file:
    ticker = line.split('-')
    if len(ticker) == 2:
        ticker, label = ticker
        #print(label.split("/"))
        sector, cname = label.split("/")[4], label.split("/")[5][:-6]
        print(ticker, sector, cname)
        ticker_list[ticker] = (sector, cname)

start = 0

for ticker, label in ticker_list.items():
    if ticker == 'YESBANK':
        break 
    start += 1
print(start)

n = 0
details = {'Balance_Sheet': 'BalanceSheetOf', 'Cash_Flow': 'CashFlowOf', 'Income_Statement': 'IncomeStatementOf',
           'Quarterly_Income_Statement': 'QuarterlyIncomeStatementOf', 'Valuation_Ratio': 'RiskPriceAndValuationOf', 
           'Profitability_Ratios': 'ProfitabilityAndManagementEfficiencyOf', 'Solvency': 'SolvencyAndFundamentalsOf',
           'Efficiency_Ratios': 'EfficiencyRatiosOf'}
for ticker, label in ticker_list.items():
    
    n += 1
    if n <= start+1:
        continue
    sector, cname= label

    print("Scrapping ticker: ", ticker)

    if not os.path.exists("scrap/"+ticker):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs("scrap/"+ticker)

    article_num = 1
    total = 0

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

    set_viewport_size(driver, 800, 600)
    for dlabel, urllabel in details.items():

        print("Detail: ", dlabel)
        filename = "scrap/"+ticker+"/"+ dlabel+".html"  
        print(filename)
        if os.path.isfile(filename):
            total += 1
            print("file exists")
            continue
        

        url = 'https://www.topstockresearch.com/INDIAN_STOCKS/'+sector+ '/'+ urllabel+cname+ '.html'

        print(url)
        driver.get(url) 
        time.sleep(2)

        w = open(filename, "w", encoding="utf-8")
        w.write(driver.page_source)
        w.close()

        
    driver.quit()