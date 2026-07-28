"""
split_data.py

Split the cleaned dataset into
training and testing sets.
"""

# ------------------------------------
# Import functions
# ------------------------------------
import pandas as pd


# ------------------------------------
# Split Function
# ------------------------------------

def split_data(df):

    # Number of rows
    total_rows = len(df)

    print(f"Total rows : {total_rows}")

    # Last 182 rows (about 6 months) for testing
    test_size = 182

    # Calculate split index
    split_index = total_rows - test_size

    # Training data
    train_df = df.iloc[:split_index].reset_index(drop=True)

    # Testing data
    test_df = df.iloc[split_index:].reset_index(drop=True)

    print()

    print(f"Training rows : {len(train_df)}")

    print(f"Testing rows  : {len(test_df)}")

    return train_df, test_df


# ------------------------------------
# Testing
# ------------------------------------

if __name__ == "__main__":

    df = pd.read_csv("processed_data_with_regime.csv")

    train_df, test_df = split_data(df)

    print()

    print("Training Data")

    print(train_df.head())

    print()

    print("Testing Data")

    print(test_df.head())

    # Save the files

    train_df.to_csv("train_data.csv", index=False)

    test_df.to_csv("test_data.csv", index=False)

    print()

    print("Files saved successfully.")