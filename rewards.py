import numpy as np

def calculate_default_reward(
    prev_portfolio_value: float,
    portfolio_value: float,
    trade_cost: float,
    peak_value: float,
    position_delta: float,
    rolling_log_rets: list
) -> float:
    """
    Reward = Sharpe-adjusted log return
           − transaction cost penalty
           − drawdown penalty
           − position reversal penalty

    This multi-component reward encourages:
    ✓ Growing the portfolio (log return)
    ✓ Doing so efficiently (Sharpe)
    ✓ Not losing too much from peaks (drawdown penalty)
    ✓ Not trading too frequently (cost penalty + reversal penalty)
    """
    if prev_portfolio_value <= 0 or portfolio_value <= 0:
        return 0.0

    # Instantaneous log return
    log_ret = np.log(portfolio_value / (prev_portfolio_value + 1e-8))

    # Rolling Sharpe component over last 20 steps
    rolling_log_rets.append(log_ret)
    if len(rolling_log_rets) > 20:
        rolling_log_rets.pop(0)

    if len(rolling_log_rets) >= 5:
        arr = np.array(rolling_log_rets)
        sharpe_component = arr.mean() / (arr.std() + 1e-8)
    else:
        sharpe_component = log_ret

    # Transaction cost penalty
    cost_penalty = (trade_cost / (portfolio_value + 1e-8))

    # Drawdown penalty (punish drawdowns from peak)
    drawdown   = (peak_value - portfolio_value) / (peak_value + 1e-8)
    dd_penalty = max(0.0, drawdown) * 0.10

    # Reversal penalty (punish flipping from long to short or vice versa)
    reversal_penalty = 0.0
    if abs(position_delta) > 1.0:
        reversal_penalty = 0.0005

    reward = sharpe_component - cost_penalty - dd_penalty - reversal_penalty
    return float(reward)


def calculate_sideways_penalty_reward(
    prev_portfolio_value: float,
    portfolio_value: float,
    trade_cost: float,
    peak_value: float,
    position_delta: float,
    rolling_log_rets: list,
    position: float
) -> float:
    """
    Reward Strategy 1:
    - Negative penalty for holding a position in a sideways market.
    - Exponential negative reward for increasing drawdown.
    """
    if prev_portfolio_value <= 0 or portfolio_value <= 0:
        return 0.0

    log_ret = np.log(portfolio_value / (prev_portfolio_value + 1e-8))
    
    # Sideways penalty: if we hold a position but the log return is extremely tiny
    sideways_penalty = 0.0
    if abs(position) > 0.1 and abs(log_ret) < 0.0005:  # e.g., less than 0.05% move
        sideways_penalty = 0.001  # small constant bleed for waiting in a flat market
        
    # Exponential Drawdown Penalty
    drawdown = (peak_value - portfolio_value) / (peak_value + 1e-8)
    exp_dd_penalty = (max(0.0, drawdown) ** 2) * 0.5  # squares the drawdown

    # Transaction cost
    cost_penalty = (trade_cost / (portfolio_value + 1e-8))

    reward = log_ret - sideways_penalty - exp_dd_penalty - cost_penalty
    return float(reward)


def calculate_sharpe_focused_reward(
    prev_portfolio_value: float,
    portfolio_value: float,
    trade_cost: float,
    peak_value: float,
    position_delta: float,
    rolling_log_rets: list,
    prev_sharpe: float
) -> tuple:
    """
    Reward Strategy 2:
    - Positive reward for increasing Sharpe ratio.
    - Negative reward for decreasing Sharpe ratio.
    - Linear negative reward for drawdown.
    
    Returns (reward, new_sharpe) so the environment can track prev_sharpe.
    """
    if prev_portfolio_value <= 0 or portfolio_value <= 0:
        return 0.0, prev_sharpe

    log_ret = np.log(portfolio_value / (prev_portfolio_value + 1e-8))
    
    rolling_log_rets.append(log_ret)
    if len(rolling_log_rets) > 20:
        rolling_log_rets.pop(0)

    current_sharpe = prev_sharpe
    if len(rolling_log_rets) >= 5:
        arr = np.array(rolling_log_rets)
        current_sharpe = arr.mean() / (arr.std() + 1e-8)

    # Reward based on Sharpe delta
    sharpe_delta = current_sharpe - prev_sharpe
    if sharpe_delta > 0:
        sharpe_reward = sharpe_delta * 0.5
    else:
        sharpe_reward = sharpe_delta * 1.0  # penalize decreases twice as hard

    # Drawdown penalty
    drawdown = (peak_value - portfolio_value) / (peak_value + 1e-8)
    dd_penalty = max(0.0, drawdown) * 0.1

    # Transaction cost
    cost_penalty = (trade_cost / (portfolio_value + 1e-8))
    
    reward = sharpe_reward - dd_penalty - cost_penalty
    return float(reward), current_sharpe
