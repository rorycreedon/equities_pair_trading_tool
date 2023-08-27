import yfinance as yf


def get_ticker_name(ticker: str):
    """
    Returns the name of the ticker.
    :param ticker: The ticker symbol.
    :return: The name of the ticker.
    """
    try:
        return yf.Ticker(ticker).info["longName"]
    except:
        return "Company name not loading from Yahoo Finance API."
