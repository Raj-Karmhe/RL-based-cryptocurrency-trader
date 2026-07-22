import numpy as np
import pandas as pd

def total_return(portfolio_values):

    initial = portfolio_values[0]

    final = portfolio_values[-1]

    return (
        (final - initial)
        / initial
    ) * 100

def maximum_drawdown(portfolio_values):

    values = np.array(portfolio_values)

    running_max = np.maximum.accumulate(values)

    drawdowns = (
        values - running_max
    ) / running_max

    return drawdowns.min() * 100

def sharpe_ratio(portfolio_values):

    values = np.array(portfolio_values)

    returns = (
        values[1:] - values[:-1]
    ) / values[:-1]


    if returns.std() == 0:
        return 0


    return (
        np.sqrt(252)
        *
        returns.mean()
        /
        returns.std()
    )

def win_rate(trades):

    if len(trades) == 0:
        return 0


    winners = [
        trade for trade in trades
        if trade["profit"] > 0
    ]

    return (
        len(winners)
        /
        len(trades)
    ) * 100

def generate_report(
    portfolio_values,
    trades
):

    report = {

        "Total Return (%)":
            total_return(
                portfolio_values
            ),

        "Max Drawdown (%)":
            maximum_drawdown(
                portfolio_values
            ),

        "Sharpe Ratio":
            sharpe_ratio(
                portfolio_values
            ),

        "Win Rate (%)":
            win_rate(
                trades
            ),

        "Number of Trades":
            len(trades)
    }

    return report
if __name__ == '__main__':
    import os
    if os.path.exists("results/evaluation_results.csv"):
        results = pd.read_csv("results/evaluation_results.csv")
        portfolio_values = results["portfolio_value"].values
        print("Performance metrics loaded from results/evaluation_results.csv:")
        print(f"Total Return (%): {total_return(portfolio_values):.2f}%")
        print(f"Max Drawdown (%): {maximum_drawdown(portfolio_values):.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio(portfolio_values):.2f}")
    else:
        print("No evaluation results found at results/evaluation_results.csv. Please run Evaluation.py first.")