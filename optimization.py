'''
Optimization utilizating forecast
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_profit(df_time_prices):
  '''
  Calculate profit for given price time series.
  Return: schedule, profit
  '''
  prices = df_time_prices.y.values
  num_steps = df_time_prices.shape[0]


  def objective_func(x):
      '''
      Cost function to optimize.
      '''
      return -np.sum(prices * x)


  def battery_constraints(x):
      '''
      Constraints: battery limits and SOC update
      '''
      soc = np.cumsum(x)  # acc SOC over time
      return np.hstack(   # limits: 0 <= SOC <= 1
                        (
                            [1 - soc[0]],
                                 soc,
                            [1 - soc[-1]]
                         )
                      )

  # charge/discharge rate limits: max 1 MWh per hour
  action_constraints = [(-1, 1) for _ in range(num_steps)]

  # solve optimization
  result = minimize(objective_func,
                    np.zeros(num_steps),
                    bounds = action_constraints,
                    constraints = {'type': 'ineq', 'fun': battery_constraints})

  # optimal schedule and profit
  opt_schedule = result.x
  opt_profit = -result.fun
  return opt_schedule, opt_profit


fcst = pd.read_csv('./fcst.csv')
df_resampled = pd.read_csv('./df_resampled.csv')


### NOTE Simplification to latest 500 time steps, due to limited available computation
ts_slicing_len = 500

# historic prices
hist_opt_schedule, hist_opt_profit = get_profit(df_resampled[:ts_slicing_len])
print('Historic prices profit:', hist_opt_profit)

# predicted prices
y_pred_df = fcst[['ds','yhat']].set_index('ds')
y_pred_df  = y_pred_df.rename(columns={'yhat':'y'})
pred_opt_schedule, pred_opt_profit = get_profit(y_pred_df[:ts_slicing_len])
print('Predicted prices profit:', pred_opt_profit)

# Strategy profit comparison
print('Prediction vs historic price strategy difference:', pred_opt_profit - hist_opt_profit)


