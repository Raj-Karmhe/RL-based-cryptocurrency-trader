import ccxt
import pandas as pd
from datetime import datetime, timedelta


def get_data():

    # ---------------------------------
    # Connect to Binance
    # ---------------------------------
    exchange = ccxt.binance()

    # ---------------------------------
    # Settings
    # ---------------------------------
    symbol = 'BTC/USDT'
    timeframe = '1d'
    limit = 1000

    # Start date (5 years ago)
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=365 * 5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    )

    all_data = []

    print("Downloading data...")

    while True:

        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if len(candles) == 0:
            break

        all_data.extend(candles)

        print(f"Downloaded {len(all_data)} candles")

        since = candles[-1][0] + 1

        if len(candles) < limit:
            break

    # ---------------------------------
    # Create DataFrame
    # ---------------------------------

    df = pd.DataFrame(
        all_data,
        columns=[
            'Timestamp',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]
    )

    # Convert timestamp into readable date
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

    return df


# -----------------------------------------------------
# This part runs ONLY if import_data.py is executed directly
# -----------------------------------------------------

if __name__ == "__main__":

    df = get_data()

    print(df.head())

    print(df.tail())

    print(df.info())

    df.to_csv("BTCUSDT_5years.csv", index=False)

    print("CSV saved successfully!")