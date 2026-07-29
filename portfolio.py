"""
portfolio.py

Handles all portfolio-related operations.

Responsibilities:
- Buy
- Sell
- Hold
- Transaction fees
- Slippage
- Portfolio valuation
"""

class Portfolio:

    def __init__(
        self,
        initial_balance=100000,
        transaction_fee=0.001,
        slippage=0.0005,
    ):

        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage

        self.reset()

    # -------------------------------------------------

    def reset(self):

        self.balance = self.initial_balance

        self.btc_held = 0.0

        self.portfolio_value = self.initial_balance

    # -------------------------------------------------

    def buy(self, fraction, price):

        if fraction <= 0:
            return

        amount = self.balance * fraction

        execution_price = price * (1 + self.slippage)

        fee = amount * self.transaction_fee

        btc = (amount - fee) / execution_price

        self.balance -= amount

        self.btc_held += btc

    # -------------------------------------------------

    def sell(self, fraction, price):

        if fraction <= 0:
            return

        btc_to_sell = self.btc_held * fraction

        execution_price = price * (1 - self.slippage)

        sale = btc_to_sell * execution_price

        fee = sale * self.transaction_fee

        self.balance += sale - fee

        self.btc_held -= btc_to_sell

    # -------------------------------------------------

    def update_value(self, current_price):

        self.portfolio_value = (

            self.balance

            +

            self.btc_held * current_price

        )

        return self.portfolio_value

    # -------------------------------------------------

    def rebalance(self, target_fraction, current_price):
        """
        Rebalance portfolio so that the desired fraction
        of total portfolio value is invested in BTC.

        target_fraction:
            0.0 -> 0% BTC
            1.0 -> 100% BTC
        """

        # Update latest portfolio value
        self.update_value(current_price)

        target_fraction = max(0.0, min(1.0, target_fraction))

        target_btc_value = self.portfolio_value * target_fraction

        current_btc_value = self.btc_held * current_price

        difference = target_btc_value - current_btc_value

        # Need to buy BTC
        if difference > 0:

            fraction_to_buy = min(
                difference / self.balance,
                1.0
            ) if self.balance > 0 else 0

            self.buy(fraction_to_buy, current_price)

        # Need to sell BTC
        elif difference < 0:

            btc_value = self.btc_held * current_price

            fraction_to_sell = min(
                abs(difference) / btc_value,
                1.0
            ) if btc_value > 0 else 0

            self.sell(fraction_to_sell, current_price)

        self.update_value(current_price)

# for testing
'''if __name__ == "__main__":

    p = Portfolio()

    print("Initial Balance:", p.balance)

    p.buy(0.5, 100000)

    print("\nAfter Buying")

    print("Balance:", p.balance)

    print("BTC:", p.btc_held)

    p.sell(0.25, 120000)

    print("\nAfter Selling")

    print("Balance:", p.balance)

    print("BTC:", p.btc_held)

    print("Portfolio:", p.update_value(120000))'''