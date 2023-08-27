import yfinance as yf
import pandas as pd
from typing import List

from database.connector import MongoDBConnector


def fetch_components() -> List:
    """
    Fetches the list of components of the S&P 500, NADSAQ 100 and Russell 2000 indices.
    :return: A list of all the tickers in the three indices.
    """
    # S&P 500
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    sp500_table = tables[0]
    sp500_tickers = sp500_table["Symbol"].tolist()

    # NASDAQ 100
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url, header=0)
    nasdaq_symbols = tables[4]["Ticker"].tolist()

    # Russell 2000
    with open("data/russell_2000_symbols.txt", "r") as f:
        russell_2000_symbols = []
        for line in f:
            line = line.strip()
            russell_2000_symbols.append(line)

    # Combine all tickers
    all_tickers = sp500_tickers + nasdaq_symbols + russell_2000_symbols
    all_tickers = list(set(all_tickers))  # drop duplicates

    # Add on the indicies themselves
    all_tickers += ["^GSPC", "^IXIC", "^RUT"]

    return all_tickers


def fetch_data(
    tickers: List, start_date: str = "2020-01-01", end_date: str = "2023-07-31"
) -> pd.DataFrame:
    """
    Fetches end of day data for given tickers from Yahoo Finance.
    :param tickers: A list of tickers to fetch data for.
    :param start_date: The start date for the data.
    :param end_date: The end date for the data.
    :return: A DataFrame with the data.
    """
    print("Downloading data from Yahoo Finance...")
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d")["Close"]
    data = data.dropna(axis=1, how="all")
    fetched_tickers = list(data.columns)
    return data, fetched_tickers


def load_all_data(start_date: str = "2020-01-01", end_date: str = "2023-07-31") -> None:
    """
    Loads data for all tickers into MongoDB.
    :return: None
    """
    # Initialize MongoDB connector
    db_connector = MongoDBConnector()

    # Get tickers
    tickers = fetch_components()

    # Download data
    data, updated_tickers = fetch_data(
        tickers=tickers, start_date=start_date, end_date=end_date
    )

    # Store data in database
    print("Storing in database...")
    list_of_collections = db_connector.list_tickers()
    for ticker in updated_tickers:
        if ticker in list_of_collections:
            db_connector.update_data(ticker, data[ticker])
        else:
            db_connector.store_data(ticker, data[ticker])

    print("Data stored in database.")


if __name__ == "__main__":
    load_all_data()
