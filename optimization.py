'''
Evaluate battery trading optimization strategy.

Maximizes profit by buying low (charging) and selling high (discharging).
Uses Linear Programming for efficient optimization.

Problem formulation:
    Maximize: -sum(price[t] * x[t])  for t = 0..T
    Where:    x[t] > 0 = charge (buy)
              x[t] < 0 = discharge (sell)

    Subject to:
        -1 <= x[t] <= 1          # Charge/discharge rate limit (1 MWh/h)
         0 <= SOC[t] <= 1        # Battery capacity limits
        SOC[t] = sum(x[0:t])     # State of charge = cumulative energy (battery starts empty)

The LP efficiently finds the optimal charge/discharge schedule given prices.
A schedule planned on forecast prices is always settled at the actual prices,
hence planning on the actual prices (perfect foresight) is the upper bound.
'''

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def get_schedule(prices):
    '''
    Calculate optimal charge/discharge schedule for given price time series using Linear Programming.

    Mathematical formulation:
        Minimize: sum(price[t] * x[t])    (= maximize profit -sum(price[t] * x[t]))
        Subject to:
            -1 <= x[t] <= 1           (action bounds)
            0 <= cumsum(x[0:t]) <= 1  (SOC constraints)
    '''
    num_steps = len(prices)

    # Objective: buying (x > 0) costs price * x, selling (x < 0) earns it
    # This is a linear objective: c^T * x where c = prices
    c = prices

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
    if not result.success:
        raise RuntimeError(f'Schedule optimization failed: {result.message}')
    return result.x


def get_profit(schedule, prices):
    '''
    Calculate trading profit of a charge/discharge schedule, settled at the given prices.
    '''
    # Selling (x < 0) earns the price, buying (x > 0) costs it
    return -np.dot(prices, schedule)


if __name__ == '__main__':
    df_resampled = pd.read_csv('./df_resampled.csv')

    ts_slicing_len = 2500
    prices = df_resampled.y.values[:ts_slicing_len]

    opt_schedule = get_schedule(prices)
    opt_profit = get_profit(opt_schedule, prices)
    print(f'Historic prices profit: {opt_profit:.2f} € (total)')
    print(f'Historic prices profit: {opt_profit / ts_slicing_len * 24:.2f} €/day (avg)')

    soc = np.cumsum(opt_schedule)
    print(f'SOC range: [{soc.min():.3f}, {soc.max():.3f}] (should be [0, 1])')
    print(f'Action range: [{opt_schedule.min():.3f}, {opt_schedule.max():.3f}] (should be [-1, 1])')
