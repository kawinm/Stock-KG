import bs4
import csv
import requests

head = {
    'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/87.0.4280.88 Safari/537.36 "
}

with open("../../data/nifty500_constituents.csv") as ticker_file:
    ticker_reader = csv.DictReader(ticker_file)

    for row in ticker_reader:
        ticker = row["Symbol"]
        print(ticker)
        retrieve_url = "https://www.nseindia.com/get-quotes/equity?symbol="+ticker
        session = requests.session()
        print(retrieve_url)
        response = session.get(retrieve_url, headers=head, timeout=15)
        print("yes")
        

        from requests_html import HTMLSession
        s = HTMLSession()
        response = s.get(retrieve_url, headers=head)
        response.html.render()
        print(response.text)
        cdsgj
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        table = soup.findAll('div', {'class': 'table-onerow'}, recursive=True)
        print(table)
        for n, td in enumerate(table):
            print(td.td)
            print(f"index {n}: {td.text.strip()}")
        
        dgk