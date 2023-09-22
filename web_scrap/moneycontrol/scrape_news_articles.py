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


# pass the defined options and service objects to initialize the web driver 
# stock- mhcp, mdlz
#ticker_list_all = ['CPRT', 'ROST', 'BIDU', 'ODFL', 'KDP', 'NFLX', 'SBUX', 'MELI', 'MRVL', 'BIIB', 'INTC', 'SNPS', 'SGEN', 'DLTR', 'INTU', 'AMZN', 'NTES', 'DDOG', 'CDNS', 'GILD', 'SIRI', 'DOCU', 'ORLY', 'AMGN', 'META', 'FISV', 'ADI', 'MU', 'BKNG', 'SWKS', 'CRWD', 'CEG', 'PCAR', 'ADP', 'AMAT', 'GOOG', 'CTAS', 'TXN', 'NVDA', 'ZM', 'ALGN', 'WDAY', 'NXPI', 'LRCX', 'TSLA', 'ASML', 'CMCSA', 'ENPH', 'PANW', 'XEL', 'ANSS', 'MTCH', 'TEAM', 'AVGO', 'MRNA', 'VRSN', 'KLAC', 'LULU', 'VRTX', 'GOOGL', 'PAYX', 'ADBE', 'ILMN', 'AAPL', 'AEP', 'CSCO', 'LCID', 'VRSK', 'AMD', 'EA', 'COST', 'EBAY', 'ISRG', 'ABNB', 'PEP', 'FAST', 'MSFT', 'ATVI', 'REGN', 'PDD', 'CSX', 'MNST', 'IDXX', 'SPLK', 'KHC', 'JD', 'QCOM', 'ZS', 'ADSK', 'WBA', 'CTSH', 'FTNT', 'DXCM', 'PYPL', 'HON', 'CHTR', 'MAR', 'TMUS', 'EXC', 'AZN']
    
#'TWTR' and 'ABMD' defunct and delisted
ticker_list = {}

ticker_file = open('ticker_map.txt', "r").readlines()
for line in ticker_file:
    ticker, label = line.split('-')
    label = label.split("/")[-1]
    print(ticker, label)
    ticker_list[ticker] = label


start = 0
#for ticker in ticker_list:
    #if ticker == 'EQIX':
    #    break 
    #start += 1
print(start)
for ticker, label in ticker_list.items():

    print("Scrapping ticker: ", ticker)

    if not os.path.exists("scrap/"+ticker):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs("scrap/"+ticker)

    article_num = 1
    for page in range(1, 500):

        print("Page number: ", page)
        filename = "scrap/"+ticker+"/page_"+str(page)+".html"  

        if not os.path.isfile(filename):
            print("file does not exists")
            continue

        source = open(filename, "r", encoding="utf-8")

        links = []
        soup = BeautifulSoup(source.read(), 'html.parser')

        
        for article in soup.find_all('article'):
            url = article.find('a', href=True)
            if url:
                link = url['href']
                link = 'https://seekingalpha.com' + link
                print(link)
                links.append(link)
                filename = "scrap/"+ticker+"/"+str(article_num)+".html"
                print(article_num)
                

                if os.path.isfile(filename):
                    article_num += 1
                    continue

                my_options = webdriver.ChromeOptions()
                #my_options.add_argument(f'user-data-dir=Scraper')
                my_options.add_argument('--enable-javascript')
                my_options.add_argument('--disable-blink-features=AutomationControlled')
                my_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                my_options.add_experimental_option('useAutomationExtension', False)
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=my_options)
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")

                driver.get(link) 
                time.sleep(5)

                w = open(filename, "w", encoding="utf-8")
                w.write(driver.page_source)
                w.close()
                
                driver.quit()
                article_num += 1

        print(article_num)
        if len(links) < 40:
            print("Page over")
            break

       

        

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