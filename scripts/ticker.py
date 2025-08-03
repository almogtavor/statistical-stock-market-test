import requests
from bs4 import BeautifulSoup

URLS = {
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
}

def fetch_tickers(url, symbol_col=0, table_index=0):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    
    tickers = []
    table = tables[table_index]  # Select specific table
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) > symbol_col:
            ticker_text = cols[symbol_col].get_text(strip=True)
            # Clean up ticker symbols - remove any extra whitespace and newlines
            ticker_text = ticker_text.replace('\n', ' ').strip()
            # Skip empty or invalid entries
            if ticker_text and len(ticker_text) <= 10 and ticker_text.replace('.', '').replace('-', '').isalnum():
                tickers.append(ticker_text)
    return tickers

if __name__ == "__main__":
    # Nasdaq-100: Table 3 (current components), column 0 (Ticker)
    nasdaq100 = fetch_tickers(URLS["nasdaq100"], symbol_col=0, table_index=3)  
    # S&P 500: Table 0 (current components), column 0 (Symbol) 
    sp500 = fetch_tickers(URLS["sp500"], symbol_col=0, table_index=0)          
    # Dow 30: Table 0 (current components), column 1 (Symbol)  
    dow30 = fetch_tickers(URLS["dow30"], symbol_col=1, table_index=0)

    print(f"Nasdaq‑100 ({len(nasdaq100)}):")
    print(nasdaq100, "\n")

    print(f"S&P 500 ({len(sp500)}):")
    print(sp500, "\n")

    print(f"Dow 30 ({len(dow30)}):")
    print(dow30)
