import pickle 
import os
import pandas as pd
import bs4 as bs

directory = "scrap/"

data = {}

for ticker_dir in os.listdir(directory):
    f = os.path.join(directory, ticker_dir)
    # checking if it is a file

    for filename in os.listdir(f):
        f2 = os.path.join(f, filename)

        if os.path.isfile(f2):
            print(f2)

            files = open(f2, "r")
            soup = bs.BeautifulSoup(files, features="lxml")
            table = soup.find_all('table', recursive=True)
            #print(table[0])
            if len(table) >= 1:
                csv = pd.read_html(str(table[0]))
                if ticker_dir not in data:
                    data[ticker_dir] = [csv]
                else:
                    data[ticker_dir] = data[ticker_dir] + [csv]

with open("financials_nse.pkl", "wb") as f:
    pickle.dump(data, f)

