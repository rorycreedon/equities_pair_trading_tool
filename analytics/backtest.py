import pandas as pd
import numpy as np
import optuna
from typing import Union

from database import MongoDBConnector

dbconnector = MongoDBConnector()


class BackTest:
    def __init__(self, ticker1: str, ticker2: str, start_date: str, end_date: str):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start_date = start_date
        self.end_date = end_date

        self.stock1_price = dbconnector.fetch_single_stock_on_time_range(
            self.ticker1, self.start_date, self.end_date
        )["close"]
        self.stock2_price = dbconnector.fetch_single_stock_on_time_range(
            self.ticker2, self.start_date, self.end_date
        )["close"]
        self.spread = self.stock1_price - self.stock2_price

        self.signal = None
        self.daily_returns = None
        self.cumulative_returns = None

    def backtest(self, window: int, z_thresh: float):
        """
        Backtest the strategy.
        :param window: The window to use for the moving average.
        :param z_thresh: The z-score at which to buy/sell the spread.
        :return: The cumulative profit.
        """
        # Compute the rolling z-score
        z_score = (
            self.spread - self.spread.rolling(window).mean()
        ) / self.spread.rolling(window).std()

        # Compute the signal
        self.signal = pd.Series(0.0, index=z_score.index)
        self.signal[z_score > z_thresh] = -1  # short
        self.signal[z_score < -z_thresh] = 1  # long
        self.signal[np.abs(z_score) < np.abs(z_thresh)] = 0
        self.signal = self.signal.shift(
            1
        )  # shift signal 1 day to avoid look-ahead bias
        self.signal.iloc[0] = 0

        # Shift prices
        previous_stock1_price = self.stock1_price.shift(1)
        previous_stock2_price = self.stock2_price.shift(1)

        # Determine position based on previous day's prices
        long_stock1 = (self.signal == 1) & (
            previous_stock1_price > previous_stock2_price
        )
        short_stock1 = (self.signal == 1) & (
            previous_stock1_price < previous_stock2_price
        )
        long_stock2 = (self.signal == -1) & (
            previous_stock1_price < previous_stock2_price
        )
        short_stock2 = (self.signal == -1) & (
            previous_stock1_price > previous_stock2_price
        )

        # Compute daily returns
        stock1_daily_returns = self.stock1_price.pct_change().fillna(0)
        stock2_daily_returns = self.stock2_price.pct_change().fillna(0)

        # Compute strategy returns for each position
        long_stock1_returns = long_stock1 * stock1_daily_returns
        short_stock1_returns = short_stock1 * -stock1_daily_returns
        long_stock2_returns = long_stock2 * stock2_daily_returns
        short_stock2_returns = short_stock2 * -stock2_daily_returns

        # Compute the total strategy returns
        self.daily_returns = (
            long_stock1_returns
            + short_stock1_returns
            + long_stock2_returns
            + short_stock2_returns
        )

        # Compute the cumulative returns
        self.cumulative_returns = (self.daily_returns + 1).cumprod()

    def find_optimal_params(self) -> dict:
        """
        Find the optimal parameters for the strategy.
        :return: The optimal window and z-score threshold.
        """

        def objective(trial):
            window = trial.suggest_int("window", 10, 100)
            z_thresh = trial.suggest_uniform("z_thresh", 0, 3)
            self.backtest(window, z_thresh)
            return self.calculate_metrics()[0]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        return study.best_params

    def calculate_metrics(self) -> Union[float, float, float, float, float]:
        """
        Calculate the metrics for the strategy.
        :return: Sharpe ratio, Sortino ratio, maximum drawdown, annualized return, annualized volatility.
        """
        # Import EFFR data (used as risk free rate)
        effr = pd.read_csv("data/EFFR.csv", index_col=0, parse_dates=True)
        effr["EFFR"] = pd.to_numeric(effr["EFFR"], errors="coerce")
        effr["EFFR"] = effr["EFFR"] / (100 * 252)
        effr = effr.loc[self.start_date : self.end_date]
        # Forward fill missing values
        effr = effr.fillna(method="ffill")

        # Excess returns
        excess_daily_returns = self.daily_returns - effr["EFFR"]

        # Compute the Sharpe ratio
        sharpe_ratio = (
            np.sqrt(252) * excess_daily_returns.mean() / excess_daily_returns.std()
        )

        # Compute the Sortino ratio
        sortino_ratio = (
            np.sqrt(252)
            * excess_daily_returns.mean()
            / excess_daily_returns[excess_daily_returns < 0].std()
        )

        # Compute the maximum drawdown percentage
        rolling_max = self.cumulative_returns.cummax()
        drawdown = self.cumulative_returns / rolling_max - 1.0
        max_drawdown = -drawdown.min()

        # Compute the annualized return
        annualized_return = (
            self.cumulative_returns.iloc[-1] ** (252 / len(self.cumulative_returns)) - 1
        )

        # Compute the annualized volatility
        annualized_volatility = self.daily_returns.std() * np.sqrt(252)

        return (
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            annualized_return,
            annualized_volatility,
        )
