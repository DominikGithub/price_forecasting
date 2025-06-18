'''
Electricity prices prohpet forecasting model.

NOTE prophet does not expect the input time series features to be standardized.
'''

import os
import multiprocessing

# optimize Prophet settings
n_cores = multiprocessing.cpu_count()
os.environ["STAN_NUM_THREADS"] = str(n_cores)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, mean_squared_error

# compatibility
import prophet
import sys


# EDA
raw_df = pd.read_csv("electricity_prices_60min.csv")
print(raw_df.head())

# drop irrelevant columns
df_clean = raw_df.drop(columns=['Currency', 'BZN|DE-LU'])

# rename columns (NOTE naming convention given by Prophet library)
df_clean = df_clean.rename(columns={ 
    'MTU (CET/CEST)':'ds',
    'Day-ahead Price [EUR/MWh]':'y'
})

# reformat the time interal formated column into standardized timestamp format. Only the intervals start timestamp will be preserved.
# The intervals start timestamp is enough to describe the interval, as the length is constant (60 minutes).

df_clean.ds = df_clean.ds.apply(lambda t: t[:16])
df_clean.ds = pd.to_datetime(df_clean.ds, format= '%d.%m.%Y %H:%M') # NOTE still in CET not UTC, irrelevant for model for now
print(df_clean.head())

# dataset statistics (keep stats for model configuration)
train_stats_df = df_clean.describe()
print(train_stats_df)

# plot raw price data
ax1 = df_clean.plot(x='ds', y='y', kind='line', figsize=(16, 5))
fig1 = ax1.get_figure()
fig1.savefig('./plots/1_raw_prices.png')


# ------------------------------------------------------------------
# Validate fundamental assumptions about the dataset

## investigate varying amount of datapoints over time
dp_density_sr = df_clean.set_index('ds').resample('1d').count()

# plot histogram of hourly (hence 24) data points per day
ax2 = dp_density_sr.plot(kind='hist', logy=True, figsize=(16, 2))
fig2 = ax2.get_figure()
fig2.savefig('./plots/2_histogram_hourly_distribution.png')


# Locate data point gap
ax3 = dp_density_sr.plot(kind='line', figsize=(16, 2))
fig3 = ax3.get_figure()
fig3.savefig('./plots/3_gap_line.png')

missing_df = df_clean.set_index('ds')\
                      .resample('1h')\
                      .asfreq()\
                      .fillna(np.nan)
# list missing time slot
print('Missing time slots:', missing_df[missing_df['y'].isnull()].shape[0])

# interpolation missing data point from previous and next neighbors
dense_df = df_clean.interpolate()

# approve there is no more left
assert dense_df[dense_df['y'].isnull()].empty, 'There are still data gaps in the data'


# revalidate dense timeseries and homogenous distribution
ax4 = dense_df.set_index('ds').resample('1d').count().plot(kind='hist', figsize=(16, 2))
fig4 = ax4.get_figure()
fig4.savefig('./plots/4_histogram_hours_per_day.png')



######################################################
# Price forecast model
######################################################

# create seasonal base model
# set prophet fit 'cap' to 100% of maximum historic price in training dataset
df_resampled = dense_df.assign(cap=train_stats_df.loc['max', 'y'])
df_resampled.to_csv('./df_resampled.csv')

# fit model
m = Prophet()       # mcmc_samples=300  -> NOTE different sampling strategy could increase the result quality, but at the cost of massive runtime complexity surge
m.fit(df_resampled)

# TODO model serialization

# config prediction
# num future intervals
future = m.make_future_dataframe(periods=20)
# max capability +10% (to consider eventual predictions beyond the training data limits up to a restricted level)
future['cap'] = train_stats_df.loc['max', 'y'] * 1.1

# predict
fcst = m.predict(future)
fcst.to_csv('./fcst.csv')

# plot prediction results
ax5 = fig = m.plot(fcst, figsize=(16, 5))
fig5 = ax5.get_figure()
fig5.savefig('./plots/5_predictions.png')

##### cross validation
df_cv = cross_validation(m, initial='120 days', period='24 hours', horizon='24 hours') 
print(df_cv)

# plot predictions (yhat) over true data (y)
ax6 = df_cv.plot(x='ds', y=['y', 'yhat'], kind='line', figsize=(16, 6))
fig6 = ax6.get_figure()
fig6.savefig('./plots/6_cross_validation.png')


###############################
# Evaluation
# MAE
mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
print(f"MAE: {mae:.2f}")
# RMSE
rmse = np.sqrt(mean_squared_error(df_cv['y'], df_cv['yhat']))
print(f"RMSE: {rmse:.2f}")
