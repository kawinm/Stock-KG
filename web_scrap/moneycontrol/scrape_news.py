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

ticker_file = open('label_map.txt', "r").readlines()
for line in ticker_file:
    ticker = line.split(' - ')
    if len(ticker) == 2:
        ticker, label = ticker
        print(ticker, label[:-1])
        ticker_list[ticker] = label[:-1]

"""

correct_ticker_label_map = {}
durl = 'https://www.moneycontrol.com/company-article/abbindia/news/'
for ticker, label in ticker_list.items():
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

    url = durl + label
    driver.get(url) 
    time.sleep(1)

    links = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    #driver.quit()
    article_num = 0
    for article in soup.find_all("div", {"class": "FL"}):
        url = article.find('a', href=True)
        if url and "sc_id=" in url["href"]:
            print(url["href"])
            labels = url["href"].split("sc_id=")[1].split("&")[0]
            print(ticker, "-", labels)
            correct_ticker_label_map[ticker] = labels
            break

for ticker, label in correct_ticker_label_map.items():
    print(ticker, "-", label)
"""
start = 0

for ticker, label in ticker_list.items():
    if ticker == 'ZENSARTECH':
        break 
    start += 1
print(start)

n = 0
years = [str(x) for x in range(2005, 2023)]
years = years[::-1]
for ticker, label in ticker_list.items():
    n += 1
    if n <= start+1:
        continue
    

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
    for year in years:
        f = 9999

        
        for page in range(1, 11):

            print("Page number: ", page)
            filename = "scrap/"+ticker+"/"+ year+ "_page_"+str(page)+".html"  
            print(filename)
            if os.path.isfile(filename):
                total += 1
                print("file exists")
                continue
            

            

            #

            url = 'https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=' + label+ '&scat=&pageno=' + str(page) +'&next=0&durationType=Y&Year=' + year + '&duration=1&news_type='

            driver.get(url) 
            #time.sleep(1)

            links = []
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            #driver.quit()
            article_num = 0
            for article in soup.find_all("div", {"class": "FL"}):
                url = article.find('a', href=True)
                if url:
                    link = url['href']
                    #print(link)
                    links.append(1)
                    #filename = "scrap/"+ticker+"/"+str(article_num)+".html"
                    #print(article_num)
                    article_num += 1
                    continue

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

            print(article_num)

            if len(links) == 11:
                f= page
                print("Page over")
                
                break

            w = open(filename, "w", encoding="utf-8")
            w.write(driver.page_source)
            w.close()

        
        if f == 1:
            break
    driver.quit()
     

        
        #print(driver.page_source)
        #try:
        #    content = driver.find_element(By.CSS_SELECTOR, "div[class*='iX-EL'")
        #except:
        #    break

       

        
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