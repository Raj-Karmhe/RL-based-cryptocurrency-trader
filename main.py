"""
main.py

Main file of the project.
"""

from import_data import get_data
from indicators import add_indicators
from preprocess import preprocess_data
from split_data import split_data

from environment import TradingEnvironment


# --------------------------
# Data Pipeline
# --------------------------

df = get_data()

df = add_indicators(df)

df = preprocess_data(df)

train_df, test_df = split_data(df)


# --------------------------
# Create Environment
# --------------------------

env = TradingEnvironment(train_df)

observation, info = env.reset()

print(observation)