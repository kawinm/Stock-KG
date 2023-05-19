import argparse
import csv
import urllib.request
import datetime
import time
from os.path import exists
import pandas as pd
import argparse

not_found = ['ANTM', 'BLL', 'BRK.B', 'BF.B', 'CERN', 'CTXS', 'DISCA', 'DISCK', 'DRE', 'FB', 'FBHS', 'INFO', 'KSU',
             'NLSN', 'NLOK', 'PBCT', 'VIAC', 'WLTW', 'XLNX', 'TWTR']

ticker_map = {}

def get_ticker_info():

    constituent_dict = {}
    with open("data/" + INDEX + "_constituents.csv") as ticker_file:
        ticker_reader = csv.DictReader(ticker_file)

        for row in ticker_reader:
            constituent_dict[row["Symbol"]] = {
                        "ticker": row["Symbol"],
                        "name"  : row["Name"],
                        "sector": row["Sector"]
                    }

    return constituent_dict

def get_unix_time(year, month, day):

    date_time = datetime.datetime(year=year, month=month, day=day)
    return int(time.mktime(date_time.timetuple()))


def download_historical_data(constituent_dict, start_time, end_time, index):

    num_companies = 1
    for company, details in constituent_dict.items():

        print("Downloading Data of: " + company, num_companies)

        # Not found in yahoo finance
        
        if company in not_found:
            continue
        
        if index == 'nifty500':
            company = company+'.NS'

        retrieve_url = "https://query1.finance.yahoo.com/v7/finance/download/" + company + \
                        "?period1=" + str(start_time) + "&period2=" + str(end_time) + \
                        "&interval=1d&events=history&includeAdjustedClose=true"
        
        name = details["name"].replace('-', '')
        save_file_name = details["ticker"] + "-" + name + ".csv"
        

        num_companies += 1

        if exists("data/"+ INDEX + "/" +save_file_name):
            print("File exists")
            continue      

        urllib.request.urlretrieve(retrieve_url, "data/"+ INDEX + "/" +save_file_name)
        print("Total Number of companies: ", num_companies)

def download_dividend_history(constituent_dict, start_time, end_time, index):

    for company, details in constituent_dict.items():

        print("Downloading Dividend Data of: " + company)

        # Not found in yahoo finance
        if company in not_found or details["ticker"] in ticker_map:
            continue

        if index == 'nifty500':
            company = company+'.NS'

        retrieve_url = "https://query1.finance.yahoo.com/v7/finance/download/" + company + \
                        "?period1=" + str(start_time) + "&period2=" + str(end_time) + \
                        "&interval=1d&events=div&includeAdjustedClose=true"
        
        name = details["name"].replace('-', '')
        save_file_name = "Dividend-"+details["ticker"] + "-" + name + ".csv"

        if exists("data/"+ INDEX + "/" +save_file_name):
            print("File exists")
            continue      

        urllib.request.urlretrieve(retrieve_url, "data/dividend_history/"+save_file_name)

def download_stocksplit_data(constituent_dict, start_time, end_time, index):

    for company, details in constituent_dict.items():

        print("Downloading Data of: " + company)

        # Not found in yahoo finance
        if company in not_found or details["ticker"] in ticker_map:
            continue

        if index == 'nifty500':
            company = company+'.NS'

        retrieve_url = "https://query1.finance.yahoo.com/v7/finance/download/" + company + \
                        "?period1=" + str(start_time) + "&period2=" + str(end_time) + \
                        "&interval=1d&events=split&includeAdjustedClose=true"
        
        name = details["name"].replace('-', '')
        save_file_name = "Split-"+details["ticker"] + "-" + name + ".csv"

        ticker_map[details["ticker"]] = 1

        if exists("data/"+ INDEX + "/" +save_file_name):
            print("File exists")
            continue      

        urllib.request.urlretrieve(retrieve_url, "data/stock_split_history/"+save_file_name)
        
def join_all_data(key):
    #key = "ticker"
    div, split, info, data = {}

    for key, value in data.items():
        print(key)
        div_df = div[key]
        split_df = split[key]
        df = pd.merge(value, split_df, on ="Date", how='left')
        df = pd.merge(df, div_df, on ="Date", how='left')
        print(div_df)
        sector, name, ticker, start, end = info[key]
        save_file_name =  "data/"+ INDEX + "/" + sector + "-" + name + "-" + ticker + "-" + start + "-" + end 
        df.to_csv(save_file_name, index=False)
        print(save_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default="sp500")
    #parser.add_argument('--window', type=int, default=10)
    #parser.add_argument('--test_size', type=float, default=0.2)

    index = ['nasdaq100', 'sp500', 'nifty500']

    for INDEX in index[2:]:
        #INDEX = parser.parse_args().index

        start_time = get_unix_time(2003, 1, 1)
        end_time   = get_unix_time(2023, 1, 1)

        constituent_dict = get_ticker_info()
        print("Number of companies in listing: ", len(constituent_dict.values()))

        download_historical_data(constituent_dict, start_time, end_time, INDEX)
        download_dividend_history(constituent_dict, start_time, end_time, INDEX)
        download_stocksplit_data(constituent_dict, start_time, end_time, INDEX)

        print("Total Companies in Nasdaq and Sp500: ", len(ticker_map.keys()))

