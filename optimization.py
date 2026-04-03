'''
Evaluate battery trading optimization strategy.

Maximizes profit by buying low (charging) and selling high (discharging).
Uses Linear Programming for efficient optimization.

Problem formulation:
    Maximize: sum(price[t] * x[t])  for t = 0..T
    Where:    x[t] > 0 = discharge (sell)
              x[t] < 0 = charge (buy)

    Subject to:
        -1 <= x[t] <= 1          # Charge/discharge rate limit (1 MWh/h)
         0 <= SOC[t] <= 1        # Battery capacity limits
        SOC[t] = sum(x[0:t])     # State of charge = cumulative energy

The LP efficiently finds the optimal charge/discharge schedule given prices.
'''

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def get_profit(df_time_prices):
    '''
    Calculate optimal profit for given price time series using Linear Programming.

    Mathematical formulation:
        Maximize: sum(price[t] * x[t])
        Subject to:
            -1 <= x[t] <= 1           (action bounds)
            0 <= cumsum(x[0:t]) <= 1  (SOC constraints)
    '''
    prices = df_time_prices.y.values
    num_steps = df_time_prices.shape[0]

    # Objective: maximize sum(price * x) = minimize -sum(price * x)
    # This is a linear objective: c^T * x where c = -prices
    c = -prices

    # Build SOC constraints as inequality: A_ub @ x <= b_ub
    # For each timestep t: 0 <= sum(x[0:t]) <= 1
    #   Lower bound:  -sum(x[0:t]) <= 0  ->  -1 for each row
    #   Upper bound:   sum(x[0:t]) <= 1  ->   1 for each row
    A_ub = np.zeros((2 * num_steps, num_steps))
    b_ub = np.zeros(2 * num_steps)
    for t in range(num_steps):
        # Upper bound: SOC[t] <= 1
        A_ub[t, :t+1] = 1
        b_ub[t] = 1
        # Lower bound: SOC[t] >= 0  ->  -SOC[t] <= 0
        A_ub[num_steps + t, :t+1] = -1
        b_ub[num_steps + t] = 0

    # Action bounds: -1 <= x[t] <= 1 (charge/discharge rate limits)
    bounds = [(-1, 1) for _ in range(num_steps)]

    # Solve LP using HiGHS solver (fast for this problem size)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        opt_schedule = result.x
        # Negate to convert back from minimization to maximization
        opt_profit = -result.fun
        return opt_schedule, opt_profit
    else:
        return None, 0.0


if __name__ == '__main__':
    fcst = pd.read_csv('./fcst.csv')
    df_resampled = pd.read_csv('./df_resampled.csv')

    ts_slicing_len = 2500

    hist_opt_schedule, hist_opt_profit = get_profit(df_resampled[:ts_slicing_len])
    print(f'Historic prices profit: {hist_opt_profit:.2f} € (total)')
    print(f'Historic prices profit: {hist_opt_profit / ts_slicing_len:.2f} €/MWh (avg)')

    y_pred_df = fcst[['ds','yhat']].set_index('ds')
    y_pred_df = y_pred_df.rename(columns={'yhat':'y'})
    pred_opt_schedule, pred_opt_profit = get_profit(y_pred_df[:ts_slicing_len])
    print(f'Predicted prices profit: {pred_opt_profit:.2f} € (total)')
    print(f'Predicted prices profit: {pred_opt_profit / ts_slicing_len:.2f} €/MWh (avg)')

    print(f'Strategy difference: {pred_opt_profit - hist_opt_profit:.2f} €')

    if hist_opt_schedule is not None:
        soc = np.cumsum(hist_opt_schedule)
        print(f'SOC range: [{soc.min():.3f}, {soc.max():.3f}] (should be [0, 1])')
        print(f'Action range: [{hist_opt_schedule.min():.3f}, {hist_opt_schedule.max():.3f}] (should be [-1, 1])')


