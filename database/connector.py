from pymongo import MongoClient
import pandas as pd
from typing import Dict
from datetime import datetime


class MongoDBConnector:
    def __init__(
        self, host: str = "localhost", port: int = 27017, db_name: str = "stock_data"
    ):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.tickers_in_db = self.db.list_collection_names()

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all tickers from the database.
        :return: A dictionary where keys are tickers and values are DataFrames with the data.
        """
        data = {}
        for ticker in self.db.list_collection_names():
            data[ticker] = pd.DataFrame(list(self.db[ticker].find())).drop(
                columns="_id"
            )
        return data

    def store_data(self, ticker: str, series: pd.Series) -> None:
        """
        Stores the given series in the database.
        :param ticker: The ticker symbol.
        :param series: The series to store.
        """
        records = [{"date": date, "close": close} for date, close in series.items()]
        collection = self.db[ticker]
        collection.insert_many(records)

    def update_data(self, ticker: str, series: pd.Series) -> None:
        """
        Updates the given series in the database.
        :param ticker: The ticker symbol.
        :param series: The series to update.
        """
        records = [{"date": date, "close": close} for date, close in series.items()]
        collection = self.db[ticker]
        for record in records:
            collection.update_one(
                {"date": record["date"]},
                {"$set": {"close": record["close"]}},
                upsert=True,
            )

    def list_tickers(self) -> list:
        """
        List all tickers available in the database.
        :return: A list of tickers.
        """
        return self.db.list_collection_names()

    def fetch_data_based_on_time_range(self, start_date: str, end_date: str) -> dict:
        """
        Fetches data from MongoDB for a given date range.
        :param start_date: Start date of the data range.
        :param end_date: End date of the data range.
        :return: Dictionary containing data for each ticker within the specified date range.
        """
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        data = {}
        for ticker in self.tickers_in_db:
            cursor = self.db[ticker].find(
                {"date": {"$gte": start_date_dt, "$lte": end_date_dt}}
            )
            df = pd.DataFrame(list(cursor))
            df.set_index("date", inplace=True)
            data[ticker] = df
        return data

    def fetch_single_stock_on_time_range(
        self, ticker: str, start_date: str, end_date: str
    ) -> dict:
        """
        Fetches data from MongoDB for a given date range for a given stock
        :param ticker: The ticker symbol.
        :param start_date: Start date of the data range.
        :param end_date: End date of the data range.
        :return: Dictionary containing data for each ticker within the specified date range.
        """
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        cursor = self.db[ticker].find(
            {"date": {"$gte": start_date_dt, "$lte": end_date_dt}}
        )
        df = pd.DataFrame(list(cursor))
        df.set_index("date", inplace=True)
        return df
