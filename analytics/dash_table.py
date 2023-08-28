import pandas as pd
import numpy as np
from numba import jit
from typing import Dict, List
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

from database import MongoDBConnector

db_connector = MongoDBConnector()


class DashTable:
    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2020-12-31",
        num_of_pairs: int = 100,
        pairs_criteria: float = 0.95,
    ):
        """
        Initialise the class
        :param start_date: The start date.
        :param end_date: The end date.
        :param num_of_pairs: The number of pairs to return in the table
        :param pairs_criteria: The criteria for selecting the pairs.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.num_of_pairs = num_of_pairs
        self.pairs_criteria = pairs_criteria

        self.data_dict = db_connector.fetch_data_based_on_time_range(
            start_date=self.start_date, end_date=self.end_date
        )  # to consider whether keeping this in memory is a good idea

        self.pairs = None

    def compute_top_correlations(self) -> None:
        """
        Computes and returns the top correlated stock pairs high than the threshold.
        Returns:
        """
        # Convert the data dictionary to a DataFrame all at once
        prices = pd.concat(list(self.data_dict.values()), axis=1)
        prices.drop("_id", axis=1, inplace=True)
        prices.columns = list(self.data_dict.keys())

        # Compute the correlations
        correlations = pd.DataFrame(
            np.corrcoef(prices.values, rowvar=False),
            columns=prices.columns,
            index=prices.columns,
        )  # numpy implementation faster than pandas
        correlations = correlations - np.eye(correlations.shape[0])
        # Only consider upper triangle to avoid duplicates
        upper_tri = correlations.where(
            np.triu(np.ones(correlations.shape), k=1).astype(bool)
        )
        correlations = upper_tri.stack()[upper_tri.stack() > self.pairs_criteria]

        self.pairs = correlations.index.values
        
        # If asking for more pairs than extracted, limit to the number of pairs extracted
        if self.pairs.shape[0] < self.num_of_pairs:
            self.num_of_pairs = self.pairs.shape[0]

    @staticmethod
    @jit(nopython=True)
    def _correlation(stock1: np.ndarray, stock2: np.ndarray) -> float:
        """
        Helper function to compute the correlation between two stocks.
        :param stock1: The first stock.
        :param stock2: The second stock.
        :return: The correlation between the two stocks.
        """
        return np.corrcoef(stock1, stock2)[0, 1]

    @staticmethod
    def _half_life(stock1, stock2):
        """
        Helper function to compute the half life of the mean reversion.
        :param stock1: The first stock.
        :param stock2: The second stock.
        :return: The half life of the mean reversion.
        """
        spread = stock1 - stock2

        # Create a lagged version of the spread
        lagged_spread = np.roll(spread, 1)
        lagged_spread[0] = spread[0]  # handle the boundary condition

        delta = spread - lagged_spread
        model = sm.OLS(delta[1:], sm.add_constant(lagged_spread[1:]))
        results = model.fit()

        alpha = results.params[1]
        half_life = -np.log(2) / alpha
        if half_life <= 0:
            half_life = 1e6  # ignore if half life is negative - to figure out later on why this happens
        return half_life

    @staticmethod
    def _cointegration_p_value(stock1, stock2):
        """
        Helper function to compute the p-value of the Engle Granger test.
        :param stock1: The first stock.
        :param stock2: The second stock.
        :return: The p-value of the Engle Granger test.
        """
        _, p_value, _ = coint(stock1, stock2)
        return p_value

    def stock_selection(self, criteria: str) -> List[List[Dict]]:
        """
        Selects the top N pairs based on the given criteria.
        :param criteria: The criteria to use.
        :return: A list with the data and columns for the table.
        """
        # Dict of helper functions
        helpers = {
            "Correlation": self._correlation,
            "Cointegration P-Value": self._cointegration_p_value,
            "Mean Reversion Half Life": self._half_life,
        }
        sort_dict = {
            "Correlation": True,
            "Cointegration P-Value": False,
            "Mean Reversion Half Life": False,
        }

        # Calculate top N pairs based on criteria
        table_data = []
        for pair in self.pairs:
            table_data.append(
                {
                    "Pair": f"{pair[0]} - {pair[1]}",
                    criteria: helpers[criteria](
                        self.data_dict[pair[0]]["close"].values,
                        self.data_dict[pair[1]]["close"].values,
                    ),
                }
            )  # make sure numpy so will work with numba

        # Setup table
        table_data = sorted(
            table_data, key=lambda x: x[criteria], reverse=sort_dict[criteria]
        )[: self.num_of_pairs]
        table_cols = [
            {"name": "Pair", "id": "Pair"},
            {
                "name": criteria,
                "id": criteria,
            },
        ]
        table = [table_data, table_cols]

        # Add other criteria
        for c in helpers.keys():
            if c != criteria:
                for i in range(self.num_of_pairs):
                    stocks = table_data[i]["Pair"].split(" - ")
                    # append in value of correlation/cointegration/half life
                    table[0][i][c] = helpers[c](
                        self.data_dict[stocks[0]]["close"].values,
                        self.data_dict[stocks[1]]["close"].values,
                    )  # make sure numpy so will work with numba

                # Update columns
                table[1].append(
                    {
                        "name": c,
                        "id": c,
                    }
                )

        return table
