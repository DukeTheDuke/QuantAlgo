"""Simple Trading Algorithm Simulator

This script simulates a basic moving average crossover trading algorithm. It
is intended for educational and research purposes only and should not be used
for real financial trading.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import requests
from io import StringIO


@dataclass
class Trade:
    index: int
    action: str
    price: float


class MovingAverageCrossoverSimulator:
    """Simulates a moving average crossover trading strategy."""

    def __init__(self, prices: List[float], short_window: int = 5, long_window: int = 20, initial_cash: float = 10000.0):
        self.prices = pd.Series(prices)
        self.short_window = short_window
        self.long_window = long_window
        self.cash = initial_cash
        self.position = 0  # Positive for long, negative for short
        self.trades: List[Trade] = []

    def generate_signals(self) -> pd.DataFrame:
        """Create a DataFrame with price, short MA, long MA, and trading signals."""
        data = pd.DataFrame({'price': self.prices})
        data['short_ma'] = data['price'].rolling(window=self.short_window, min_periods=1).mean()
        data['long_ma'] = data['price'].rolling(window=self.long_window, min_periods=1).mean()
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
        return data

    def run(self):
        """Execute the simulation across the dataset."""
        data = self.generate_signals()
        for idx, row in data.iterrows():
            signal = row['signal']
            price = row['price']
            if signal == 1 and self.position <= 0:
                # Buy signal
                self.trades.append(Trade(idx, 'BUY', price))
                self.position = 1
            elif signal == -1 and self.position >= 0:
                # Sell signal
                self.trades.append(Trade(idx, 'SELL', price))
                self.position = -1
        return data

    def summary(self) -> str:
        """Return a summary of executed trades."""
        lines = [f"Starting cash: {self.cash}"]
        for trade in self.trades:
            lines.append(f"{trade.index}: {trade.action} @ {trade.price:.2f}")
        lines.append(f"Ending position: {self.position}")
        return "\n".join(lines)


def generate_random_prices(n: int = 100, seed: int = None) -> List[float]:
    """Generate synthetic price data using a noisy sine wave."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, n)
    noise = rng.normal(0, 0.5, n)
    return list(np.sin(x) + noise + 10)


def fetch_real_prices(symbol: str, limit: int = 100) -> List[float]:
    """Fetch recent closing prices for the given symbol using Stooq."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to fetch data: {exc}") from exc
    df = pd.read_csv(StringIO(response.text))
    return df['Close'].tail(limit).tolist()


def main():
    """Entry point for running the demo simulation with real data."""
    try:
        prices = fetch_real_prices("AAPL", limit=200)
    except RuntimeError as exc:
        print(exc)
        print("Falling back to random prices.\n")
        prices = generate_random_prices(200, seed=42)
    sim = MovingAverageCrossoverSimulator(prices, short_window=5, long_window=20)
    sim.run()
    print(sim.summary())


if __name__ == "__main__":
    main()
