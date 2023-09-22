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

import pickle


# pass the defined options and service objects to initialize the web driver 
# stock- mhcp, mdlz
#ticker_list_all = ['CPRT', 'ROST', 'BIDU', 'ODFL', 'KDP', 'NFLX', 'SBUX', 'MELI', 'MRVL', 'BIIB', 'INTC', 'SNPS', 'SGEN', 'DLTR', 'INTU', 'AMZN', 'NTES', 'DDOG', 'CDNS', 'GILD', 'SIRI', 'DOCU', 'ORLY', 'AMGN', 'META', 'FISV', 'ADI', 'MU', 'BKNG', 'SWKS', 'CRWD', 'CEG', 'PCAR', 'ADP', 'AMAT', 'GOOG', 'CTAS', 'TXN', 'NVDA', 'ZM', 'ALGN', 'WDAY', 'NXPI', 'LRCX', 'TSLA', 'ASML', 'CMCSA', 'ENPH', 'PANW', 'XEL', 'ANSS', 'MTCH', 'TEAM', 'AVGO', 'MRNA', 'VRSN', 'KLAC', 'LULU', 'VRTX', 'GOOGL', 'PAYX', 'ADBE', 'ILMN', 'AAPL', 'AEP', 'CSCO', 'LCID', 'VRSK', 'AMD', 'EA', 'COST', 'EBAY', 'ISRG', 'ABNB', 'PEP', 'FAST', 'MSFT', 'ATVI', 'REGN', 'PDD', 'CSX', 'MNST', 'IDXX', 'SPLK', 'KHC', 'JD', 'QCOM', 'ZS', 'ADSK', 'WBA', 'CTSH', 'FTNT', 'DXCM', 'PYPL', 'HON', 'CHTR', 'MAR', 'TMUS', 'EXC', 'AZN']

# From DLR to LAB PC
#'TWTR' and 'ABMD' defunct and delisted
ticker_list = ['MNST', 'GNRC', 'SLB', 'PWR', 'RCL', 'UAL', 'EQIX', 'INTC', 'DOV', 'IP', 'FOXA', 'AIZ', 'SYK', 'ADSK', 'QRVO', 
               'ALLE', 'CARR', 'DXC', 'HON', 'NTRS', 'MPC', 'ROP', 'FTNT', 'EVRG', 'IFF', 'SO', 'AEP', 'MLM', 
               'FIS', 'F', 'SWKS', 'SBAC', 'AON', 'MET', 'ODFL', 'BAC', 'GLW', 'RL', 'HES', 'WAT', 'RSG', 'BAX', 
               'MDLZ', 'RF', 'SPGI', 'LNT', 'CMG', 'ILMN', 'LYV', 'VRSK', 'ALGN', 'MS', 'AMGN', 'XOM', 'ROK', 
               'REGN', 'LEN', 'HIG', 'DE', 'POOL', 'LMT', 'WHR', 'DRI', 'EOG', 'FRT', 'BBWI', 'CDNS', 'NOC', 
               'CAT', 'DD', 'HSY', 'WYNN', 'WELL', 'FISV', 'NI', 'NWS', 'EMR', 'DUK', 'ECL', 'ABC', 'APD', 
               'MKC', 'BKNG', 'SEE', 'EQR', 'LYB', 'ISRG', 'UAA', 'JCI', 'LHX', 'INCY', 'SHW', 'NXPI', 'HPQ', 
               'MCD', 'ATO', 'SJM', 'TFX', 'DLR', 'ADI', 'LUMN', 'GWW', 'UHS', 'DHR', 'MRNA', 'FFIV', 'ITW', 'K', 
               'MMC', 'MSCI', 'UPS', 'CTAS', 'GOOGL', 'ORCL', 'UNH', 'UA', 'JKHY', 'PNW', 'KMB', 'CE', 'PTC', 
               'MAA', 'CVX', 'NCLH', 'COST', 'VRTX', 'CPRT', 'IQV', 'SNPS', 'EW', 'BIO', 'LNC', 'WRB', 'AEE', 
               'ANSS', 'SPG', 'CINF', 'C', 'DXCM', 'KEYS', 'ADP', 'HBI', 'IBM', 'HPE', 'DIS', 'EXR', 'ZION', 'WU', 
               'ALK', 'EA', 'LKQ', 'TSN', 'ACN', 'STZ', 'TXN', 'CF', 'UNP', 'MTCH', 'OGN', 'FRC', 'MPWR', 'PSA', 
               'MTB', 'PNR', 'NVR', 'HRL', 'PLD', 'L', 'CI', 'EFX', 'MMM', 'FOX', 'EXC', 'TAP', 'HII', 'KIM', 
               'NDAQ', 'WMB', 'DGX', 'SBUX', 'WY', 'TRV', 'PCAR', 'JBHT', 'KLAC', 'TECH', 'CHTR', 'APTV', 'VTRS', 
               'NKE', 'ULTA', 'TROW', 'CTRA', 'A', 'IEX', 'AAPL', 'NEM', 'URI', 'GS', 'NRG', 'V', 'ETR', 'HST', 
               'AMT', 'ABT', 'AVB', 'EL', 'GIS', 'BKR', 'FMC', 'PRU', 'WMT', 'DG', 'RTX', 'RJF', 'CMI', 'KEY', 
               'MDT', 'COP', 'PPG', 'MAS', 'HSIC', 'ORLY', 'GPS', 'O', 'NOW', 'DPZ', 'MCK', 'DHI', 'BLK', 'STE', 
               'ROL', 'VLO', 'ZBH', 'PG', 'ANET', 'WST', 'MRK', 'AVGO', 'LDOS', 'NVDA', 'MCHP', 'CTLT', 'AMCR', 
               'BXP', 'PNC', 'HCA', 'CRM', 'PENN', 'LLY', 'BDX', 'IT', 'APH', 'ADBE', 'DLTR', 'LOW', 'BK', 'IPGP', 
               'TEL', 'CPB', 'VFC', 'GE', 'AKAM', 'SCHW', 'OTIS', 'EXPD', 'PFG', 'AJG', 'CCL', 'PH', 'DAL', 'LRCX', 
               'CAH', 'VTR', 'TDY', 'BRO', 'CBRE', 'VRSN', 
               'CHRW', 'PEAK', 'STX', 'AXP', 'CNC', 'EBAY', 'MO', 'CNP', 'XRAY', 'FTV', 'WM', 'BWA', 'ZTS', 'NFLX', 
               'ENPH', 'LEG', 'MRO', 'ROST', 'PAYC', 'BR', 'RE', 'BMY', 'MSI', 'CMA', 'HLT', 'TPR', 'AES', 'KR', 'CSCO', 
               'CMS', 'GPC', 'CME', 'JPM', 'FE', 'MHK', 'WEC', 'ED', 'HAS', 'AIG', 'PPL', 'ALL', 'IDXX', 'MAR', 'ZBRA', 
               'FDX', 'DISH', 'KMI', 'LIN', 'TYL', 'HOLX', 'CRL', 'PAYX', 'SNA', 'CAG', 'HUM', 'INTU', 'SYF', 'SIVB', 'D', 
               'VMC', 'TDG', 'HBAN', 'CDW', 'QCOM', 'AFL', 'JNPR', 'CFG', 'AZO', 'RMD', 'IPG', 'AMD', 'ARE', 'GM', 'CCI', 
               'WRK', 'PYPL', 'AAL', 'STT', 'CLX', 'GL', 'TMO', 'TSCO', 'PVH', 'DOW', 'NTAP', 'PXD', 'HAL', 'PHM', 'KMX', 
               'ALB', 'IRM', 'RHI', 'TER', 'ES', 'DFS', 'CTVA', 'DVN', 'USB', 'PSX', 'GILD', 'AOS', 'CVS', 'CHD', 'WAB', 
               'CMCSA', 'FLT', 'TJX', 'EXPE', 'OKE', 'FAST', 'XEL', 'IVZ', 'AWK', 'MOS', 'TMUS', 'GPN', 'CDAY', 'OMC', 'VZ', 
               'KHC', 'BBY', 'MCO', 'WDC', 'PKI', 'LH', 'UDR', 'LUV', 'MSFT', 'T', 'TSLA', 'PKG', 'PM', 'COF', 'TTWO', 'ESS', 
               'TXT', 'BIIB', 'CSX', 'AAP', 'LW', 'ABBV', 'PEG', 'ICE', 'ETN', 'J', 'AMZN', 'CB', 'FCX', 'MA', 'EMN', 'YUM', 
               'NEE', 'GOOG', 'EIX', 'DTE', 'BEN', 'CBOE', 'ADM', 'AMAT', 'TRMB', 'TFC', 'CTSH', 'REG', 'PEP', 'MU', 'AMP', 
               'FITB', 'PFE', 'NWSA', 'ATVI', 'GRMN', 'SYY', 'MGM', 'COO', 'OXY', 'BA', 'XYL', 'CZR', 'NWL', 'MTD', 'AME', 
               'SWK', 'APA', 'CL', 'JNJ', 'SRE', 'WBA', 'NUE', 'BSX', 'HWM', 'AVY', 'KO', 'ETSY', 'GD', 'TT', 'IR', 'NSC', 
               'TGT', 'DVA', 'LVS', 'PGR', 'WFC', 'MKTX', 'FANG', 'HD', 'VNO', 'BIDU', 'KDP', 'MELI', 'MRVL', 'SGEN', 'NTES', 
               'SIRI', 'DOCU', 'META', 'CEG', 'CRWD', 'WDAY', 'ASML', 'ZM', 'PANW', 'TEAM', 'LULU', 'LCID', 'ABNB', 'DDOG', 
               'PDD', 'SPLK', 'JD', 'ZS', 'AZN']

ticker_list_new = ['SBUX', 'VRSK', 'ADBE', 'NVDA', 'AMGN', 'ISRG', 'KLAC', 'AAPL', 'REGN', 'BIDU', 'QCOM', 'XEL', 
                   'MNST', 'CTSH', 'EA', 'WBA', 'ADSK', 'EBAY', 'ILMN', 'NTES', 'SWKS', 'SPLK', 'ADP', 'MU', 'AMZN', 
                   'MCHP', 'NXPI', 'FTNT', 'SGEN', 'TXN', 'ENPH', 'DXCM', 'AZN', 'VRSN', 'MTCH', 'CHTR', 'AMAT', 'FISV', 
                   'TMUS', 'MSFT', 'CTAS', 'IDXX', 'TSLA', 'MAR', 'AEP', 'ALGN', 'MELI', 'GILD', 'SIRI', 
                   'EXC', 'CSCO', 'LULU', 'GOOGL', 'MRVL', 'ASML', 'BIIB', 'GOOG', 'BKNG', 'FAST', 'ORLY', 'ROST', 'VRTX', 
                   'PCAR', 'CMCSA', 'PEP', 'ATVI', 'AMD', 'ANSS', 'AVGO', 'INTU', 'SNPS', 'COST', 'DLTR', 'PANW', 'HON', 'INTC', 
                   'PAYX', 'CSX', 'LRCX', 'ADI', 'MDLZ', 'KDP', 'CPRT', 'ODFL', 'META', 'APH', 'PG', 'OMC', 'AMCR', 'KMX', 'STE', 
                   'SPG', 'GE', 'TMO', 'PNR', 'EIX', 'SCHW', 'EQR', 'GIS', 'TRMB', 'FCX', 'UPS', 'FDX', 'SYY', 'COF', 'LEN', 
                   'WELL', 'HSIC', 'BDX', 'BEN', 'LLY', 'VFC', 'WFC', 'WHR', 'UNP', 'FMC', 'GPC', 'ABT', 'IVZ', 'WAT', 'WEC', 
                   'ROK', 'BRO', 'LNC', 'AME', 'TDY', 'DAL', 'CTRA', 'ROP', 'PENN', 'UNH', 'DISH', 'DHI', 'BBWI', 'JPM', 'NWL', 
                   'A', 'PFG', 'ORCL', 'NEM', 'HBI', 'AMT', 'RE', 'ED', 'TDG', 'ARE', 'PHM', 'PEG', 'CME', 'ES', 'AES', 'GRMN', 
                   'PNW', 'MO', 'ICE', 'MRO', 'PLD', 'DOV', 'BSX', 'VLO', 'PPL', 'DFS', 'DUK', 'MAS', 'ZBH', 'C', 'LEG', 'WU', 
                   'IEX', 'LVS', 'CRL', 'EXPE', 'IFF', 'DVN', 'CNC', 'WRB', 'VTRS', 'CNP', 'MS', 'LNT', 'CAT', 'GNRC', 'NTAP', 
                   'KEY', 'PWR', 'PRU', 'HES', 'DD', 'F', 'MLM', 'WST', 'MPWR', 'APTV', 'IRM', 'NVR', 'JNJ', 'AKAM', 'LKQ', 'REG', 
                   'SEE', 'WYNN', 'PVH', 'DPZ', 'ACN', 'HSY', 'LUV', 'MHK', 'TJX', 'GWW', 'UHS', 'MSI', 'BK', 'ROL', 'AFL', 'EXR', 
                   'LH', 'NSC', 'VTR', 'BBY', 'XRAY', 'GD', 'CVX', 'HST', 'WMB', 'SIVB', 'HBAN', 'MKC', 'GM', 'TGT', 'AIZ', 
                   'IT', 'MCD', 'IPGP', 'BAX', 'NKE', 'MGM', 'IBM', 'CAH', 'RF', 'FRT', 'VZ', 'WM', 'TFC', 'AVB', 'MOS', 'TER', 
                   'CCL', 'MKTX', 'CMI', 'AJG', 'PSA', 'MA', 'PPG', 'UAL', 'XOM', 'ALB', 'EMR', 'EQIX', 'CI', 'CE', 'AWK', 'TSN', 
                   'ESS', 'FLT', 'SNA', 'AON', 'LDOS', 'AAL', 'BWA', 'TPR', 'EMN', 'SO', 'DE', 'PM', 'DXC', 'SHW', 'BAC', 'DRI', 
                   'ZION', 'CMS', 'PKG', 'PKI', 'NI', 'ALL', 'SJM', 'AIG', 'LYV', 'GLW', 'SYK', 'DLR', 'UDR', 'AXP', 'PH', 'TT', 
                   'EVRG', 'WY', 'ETN', 'NUE', 'PGR', 'LMT', 'RJF', 'V', 'RSG', 'COP', 'JBHT', 'PTC', 'SLB', 'HPQ', 'GPN', 'FFIV', 
                   'CMG', 'BKR', 'ZBRA', 'CLX', 'EXPD', 'TSCO', 'SRE', 'AVY', 'LYB', 'RTX', 'MPC', 'APD', 'EW', 'PFE', 'EOG', 
                   'L', 'COO', 'USB', 'FITB', 'DHR', 'CINF', 'TTWO', 'MAA', 'GL', 'RMD', 'CMA', 'HUM', 'BA', 'PNC', 'ULTA', 'GPS', 
                   'NTRS', 'BR', 'K', 'UAA', 'HRL', 'ALK', 'MET', 'MCK', 'RL', 'BLK', 'CBRE', 'CL', 'WDC', 'LHX', 'KMI', 'PEAK', 
                   'TFX', 'CVS', 'KR', 'VMC', 'POOL', 'TAP', 'LOW', 'DIS', 'HD', 'AMP', 'TXT', 'MTD', 'MDT', 'FIS', 'FRC', 'CCI', 
                   'KIM', 'D', 'NEE', 'CBOE', 'TROW', 'IP', 'EL', 'SBAC', 'JNPR', 'GS', 'HAS', 'LUMN', 'PXD', 'MMM', 'HII', 'SWK', 
                   'ATO', 'JCI', 'TYL', 'CAG', 'CF', 'ECL', 'CRM', 'EFX', 'YUM', 'DG', 'DTE', 'DGX', 'STT', 'CB', 'TECH', 'SPGI', 
                   'WMT', 'VNO', 'URI', 'XYL', 'MCO', 'JKHY', 'KO', 'ITW', 'HIG', 'FE', 'NOC', 'CHRW', 'PSX', 'KMB', 'HCA', 'AEE', 
                   'HAL', 'RHI', 'HOLX', 'MSCI', 'CPB', 'BMY', 'T', 'NRG', 'J', 'LIN', 'WAB', 'TEL', 'NOW', 'OXY', 'BXP', 'AZO', 
                   'ABC', 'APA', 'OKE', 'AAP', 'CHD', 'O', 'INCY', 'NDAQ', 'TRV', 'STX', 'RCL', 'ADM', 'ETR', 'STZ', 'MRK', 'MMC', 
                   'MTB', 'AOS', 'IPG', 'DVA', 'BIO', 'NFLX', 'CDNS',]

done = ['SBUX', 'VRSK', 'ADBE', 'NVDA', 'AMGN', 'ISRG', 'KLAC', 'AAPL', 'REGN', 'BIDU', 'QCOM', 'XEL', 'MNST', 'CTSH', 'EA', 'WBA', 'ADSK',
        'EBAY', 'ILMN', 'NTES', 'SWKS', 'SPLK',  'ADP', 'MU', 'AMZN', 'MCHP', 'NXPI', 'FTNT', 'SGEN', 'TXN', 'ENPH', 'DXCM', 'AZN', 'VRSN', 'MTCH',
        'CHTR', 'AMAT', 'FISV', 'TMUS', 'MSFT', 'CTAS', 'IDXX', 'TSLA', 'MAR', 'AEP', 'ALGN', 'MELI', 'GILD', 'SIRI', 'EXC', 'CSCO', 'LULU', 'GOOGL', 
        'MRVL', 'ASML', 'BIIB',]

def set_viewport_size(driver, width, height):
    window_size = driver.execute_script("""
        return [window.outerWidth - window.innerWidth + arguments[0],
          window.outerHeight - window.innerHeight + arguments[1]];
        """, width, height)
    driver.set_window_size(*window_size)


news_headline_json = {}
for ticker in ticker_list_new:

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
            break

        source = open(filename, "r", encoding="utf-8")

        links = []
        soup = BeautifulSoup(source.read(), 'html.parser')

        #driver.quit()
        article_num = 0
        for article in soup.find_all('article'):

            url = article.find('a', href=True)
            date = article.findAll('span')[1]
            if url:
                text = url.get_text()
                news_id = url['href'].split('/')[2].split('-')[0]
                
                news_headline_json[ticker] = news_headline_json.get(ticker, []) + [[news_id, date.text, text]]
                links.append(1)
                #filename = "scrap/"+ticker+"/"+str(article_num)+".html"
                


                

        
        #print(driver.page_source)
        #try:
        #    content = driver.find_element(By.CSS_SELECTOR, "div[class*='iX-EL'")
        #except:
        #    break

with open("news_headlines.pkl", "wb") as f:
    pickle.dump(news_headline_json, f)


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