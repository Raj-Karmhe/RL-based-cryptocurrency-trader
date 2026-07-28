import ta

from import_data import get_data


# ----------------------------
# Function to add indicators
# ----------------------------

def add_indicators(df):

    # ----------------------------
    # Trend Indicators
    # ----------------------------

    df["SMA20"] = ta.trend.sma_indicator(
        close=df["Close"],
        window=20
    )

    df["SMA50"] = ta.trend.sma_indicator(
        close=df["Close"],
        window=50
    )

    df["EMA20"] = ta.trend.ema_indicator(
        close=df["Close"],
        window=20
    )

    # ----------------------------
    # Momentum
    # ----------------------------

    df["RSI"] = ta.momentum.rsi(
        close=df["Close"],
        window=14
    )

    # ----------------------------
    # MACD
    # ----------------------------

    df["MACD"] = ta.trend.macd(
        close=df["Close"]
    )

    df["MACD_SIGNAL"] = ta.trend.macd_signal(
        close=df["Close"]
    )

    # ----------------------------
    # Bollinger Bands
    # ----------------------------

    bb = ta.volatility.BollingerBands(
        close=df["Close"],
        window=20,
        window_dev=2
    )

    df["BB_UPPER"] = bb.bollinger_hband()

    df["BB_LOWER"] = bb.bollinger_lband()

    # ----------------------------
    # ATR
    # ----------------------------

    df["ATR"] = ta.volatility.average_true_range(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    # ----------------------------
    # Volume MA
    # ----------------------------

    df["VOLUME_MA20"] = df["Volume"].rolling(20).mean()

    return df


# ----------------------------
# Test the function
# ----------------------------

if __name__ == "__main__":

    df = get_data()

    df = add_indicators(df)

    print(df.head(30))

    print()

    print(df.columns)

    print()

    print(df.shape)