"""
preprocess.py

This file cleans the dataset before it is
used for training.
"""

# -----------------------------------
# Import functions
# -----------------------------------

from import_data import get_data
from indicators import add_indicators


# -----------------------------------
# Preprocessing Function
# -----------------------------------

def preprocess_data(df):

    print("Shape before preprocessing:")
    print(df.shape)

    # -----------------------------------
    # Remove duplicate rows
    # -----------------------------------

    df = df.drop_duplicates()

    # -----------------------------------
    # Remove rows containing NaN
    # -----------------------------------

    df = df.dropna()

    # -----------------------------------
    # Reset index
    # -----------------------------------

    df = df.reset_index(drop=True)

    print()

    print("Shape after preprocessing:")
    print(df.shape)

    print()

    print("Missing values in each column:")

    print(df.isnull().sum())

    return df


# -----------------------------------
# Testing
# -----------------------------------

if __name__ == "__main__":

    df = get_data()

    df = add_indicators(df)

    df = preprocess_data(df)

    print()

    print(df.head())