"""
Calculate trading profit for different price forecasting models.

Compares battery arbitrage profit across all forecasting models.

Usage:
    python calculate_profit.py  # Compare all models
    python calculate_profit.py  # Custom sample size
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os


def get_profit(prices):
    """
    Calculate optimal profit using Linear Programming.

    Maximize: sum(price[t] * x[t])
    Subject to: -1 <= x[t] <= 1, 0 <= cumsum(x[0:t]) <= 1
    """
    num_steps = len(prices)
    c = -prices

    A_ub = np.zeros((2 * num_steps, num_steps))
    b_ub = np.zeros(2 * num_steps)
    for t in range(num_steps):
        A_ub[t, :t+1] = 1
        b_ub[t] = 1
        A_ub[num_steps + t, :t+1] = -1

    bounds = [(-1, 1) for _ in range(num_steps)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return -result.fun if result.success else 0.0


def load_predictions(model_name):
    """Load predictions for a given model."""
    if model_name == 'prophet':
        return pd.read_csv('./fcst.csv')['yhat'].values
    elif model_name in ['catboost', 'xgboost', 'lightgbm']:
        pred_file = f'./predictions_{model_name}.csv'
        if os.path.exists(pred_file):
            return pd.read_csv(pred_file)['y'].values
        print(f"Warning: {pred_file} not found. Run tree_forecasting.py first.")
        return None
    return None


def compare_profit():
    n_samples = 600
    print("="*60)
    print("BATTERY ARBITRAGE PROFIT CALCULATION")
    print("="*60)

    df_resampled = pd.read_csv('./df_resampled.csv')
    df_test = df_resampled[df_resampled['ds'] >= '2022-06-01']
    actual_prices = df_test['y'].values[:n_samples]

    hist_profit = get_profit(actual_prices)

    print(f"\nTest period: {df_test['ds'].iloc[0]} to {df_test['ds'].iloc[n_samples-1]}")
    print(f"Samples: {n_samples}\n")

    print(f"{'Model':<15} {'Profit (€)':>12} {'€/MWh':>10} {'vs Hist':>10}")
    print("-" * 50)
    print(f"{'Perfect Foresight':<15} {hist_profit:>12.2f} {hist_profit/n_samples:>10.2f} {'-':>10}")

    models = ['catboost', 'xgboost', 'lightgbm', 'prophet']
    for model_name in models:
        predictions = load_predictions(model_name)
        if predictions is None:
            continue
        predictions = predictions[:n_samples]
        profit = get_profit(predictions)
        vs_hist = ((profit / hist_profit) - 1) * 100 if hist_profit > 0 else 0
        print(f"{model_name.capitalize():<15} {profit:>12.2f} {profit/n_samples:>10.2f} {vs_hist:>+9.1f}%")

    print("\nPrice Statistics:")
    print(f"{'Model':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*60)

    all_prices = {'actual': actual_prices}
    for model_name in models:
        predictions = load_predictions(model_name)
        if predictions is not None:
            all_prices[model_name] = predictions[:n_samples]

    for name, prices in all_prices.items():
        print(f"{name.capitalize():<15} {np.mean(prices):>10.1f} {np.std(prices):>10.1f} "
              f"{np.min(prices):>10.1f} {np.max(prices):>10.1f}")
    print("\n" + "="*60)


if __name__ == '__main__':
    compare_profit()
