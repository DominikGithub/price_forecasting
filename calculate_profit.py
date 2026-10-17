"""
Calculate trading profit for different price forecasting models.

Compares battery arbitrage profit across the tree based forecasting models.
Each charge/discharge schedule is planned on the forecast prices, but its
profit is settled at the actual prices. Perfect foresight (planning on the
actual prices) is therefore the upper bound for every forecast.

Usage:
    python calculate_profit.py              # Compare models with 24h sample length and 24h horizon
    python calculate_profit.py -l 6 -z 1    # Compare models of another tree_forecasting.py setting
"""

import argparse
import os
import numpy as np
import pandas as pd
from optimization import get_schedule, get_profit


def load_predictions(model_name, length, horizon, features):
    """Load test set predictions of a tree model (saved by tree_forecasting.py), indexed by target time."""
    pred_file = f'./predictions_{model_name}_l{length}_z{horizon}_af{features}.csv'
    if os.path.exists(pred_file):
        return pd.read_csv(pred_file, parse_dates=['ds']).set_index('ds')['y_pred']
    print(f"Warning: {pred_file} not found. Run tree_forecasting.py first.")
    return None


def compare_profit(length, horizon):
    print("="*60)
    print("BATTERY ARBITRAGE PROFIT CALCULATION")
    print("="*60)

    prices = pd.read_csv('./df_resampled.csv', parse_dates=['ds']).set_index('ds')['y']

    forecasts = {}
    for features in [False, True]:
        for model_name in ['catboost', 'xgboost', 'lightgbm']:
            predictions = load_predictions(model_name, length, horizon, features)
            if predictions is not None:
                forecasts[model_name.capitalize() + (' +AF' if features else '')] = predictions
    if not forecasts:
        return

    # common test period of all models
    fcst_df = pd.concat(forecasts, axis=1).dropna()
    # naive baseline: price of the same hour on the previous day
    fcst_df['Naive (prev. day)'] = prices.shift(24, freq='h')
    actual_prices = prices.loc[fcst_df.index].values
    n_days = len(actual_prices) / 24

    print(f"\nTest period: {fcst_df.index[0]} to {fcst_df.index[-1]} ({n_days:.0f} days)")
    print("Schedules are planned on the forecast prices and settled at the actual prices.\n")

    perfect_schedule = get_schedule(actual_prices)
    perfect_profit = get_profit(perfect_schedule, actual_prices)

    print(f"{'Model':<18} {'Profit (€)':>11} {'€/day':>8} {'% Perfect':>10} {'MAE':>7} {'Cycles/day':>11} {'Expected (€)':>13}")
    print("-" * 83)
    print(f"{'Perfect foresight':<18} {perfect_profit:>11.2f} {perfect_profit/n_days:>8.2f} {100:>9.1f}% {'-':>7} "
          f"{np.abs(perfect_schedule).sum()/2/n_days:>11.2f} {perfect_profit:>13.2f}")

    for name in fcst_df:
        forecast = fcst_df[name].values
        schedule = get_schedule(forecast)
        profit = get_profit(schedule, actual_prices)
        expected_profit = get_profit(schedule, forecast)
        mae = np.mean(np.abs(forecast - actual_prices))
        print(f"{name:<18} {profit:>11.2f} {profit/n_days:>8.2f} {100*profit/perfect_profit:>9.1f}% {mae:>7.2f} "
              f"{np.abs(schedule).sum()/2/n_days:>11.2f} {expected_profit:>13.2f}")

    print("\nExpected (€): profit the schedule would make if the forecast prices were exact (not achievable).")

    print("\nPrice Statistics:")
    print(f"{'Model':<18} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*62)

    all_prices = {'Actual': actual_prices} | {name: fcst_df[name].values for name in fcst_df}
    for name, values in all_prices.items():
        print(f"{name:<18} {np.mean(values):>10.1f} {np.std(values):>10.1f} "
              f"{np.min(values):>10.1f} {np.max(values):>10.1f}")
    print("\n" + "="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Profit_calculation')
    parser.add_argument('-l', "--length", type=int, default=24)
    parser.add_argument('-z', "--horizon", type=int, default=24)
    args = parser.parse_args()
    compare_profit(args.length, args.horizon)
