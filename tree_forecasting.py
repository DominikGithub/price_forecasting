'''
Electricity prices tree based forecasting model.

NOTE in contrast to Prohpet library, the XGBoost models input features do require standardization.
Targets are not standardized.
'''

import os
import multiprocessing as mp
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

N_STEPS_PRED_HORIZON = 24
N_STEPS_SAMPLE_LEN = 24


# load data
raw_df = pd.read_csv("electricity_prices_60min.csv")
print(raw_df)

# drop and rename columns
clean_df = raw_df.drop(columns=['Currency', 'BZN|DE-LU'])
clean_df = clean_df.rename(columns={ 
    'MTU (CET/CEST)':'ds',
    'Day-ahead Price [EUR/MWh]':'y'
})
# clean date time column
clean_df.ds = clean_df.ds.apply(lambda t: t[:16])
clean_df.ds = pd.to_datetime(clean_df.ds, format= '%d.%m.%Y %H:%M')
print(clean_df)

##### Create prediction target values
# Shift prediction values by 24hours
clean_df = clean_df.set_index('ds')

# NOTE data cleansing by interpolating missing gaps
clean_df = clean_df.interpolate()

# create prediction values
clean_df['y_target'] = clean_df.y.shift(periods=N_STEPS_PRED_HORIZON, freq="h")
# fill empty target values
clean_df.loc[clean_df['y_target'].isna(), 'y_target'] = clean_df.loc[clean_df['y_target'].isna(), 'y']

#### cross valdiation split NOTE last month June as test dataset
train_df = clean_df.loc[clean_df.index < '2022-06-01 00:00:00']
test_df = clean_df.loc[clean_df.index >= '2022-06-01 00:00:00']

n_test_samples = test_df.shape[0]
print('# test samples:', n_test_samples)


#### alternative sk build in series cv split
# ts_cv = TimeSeriesSplit(
#     n_splits=2,
#     gap=24,
#     # max_train_size=10000,
#     test_size=n_test_samples,
# )
# all_splits = list(ts_cv.split(clean_df['y'].values, clean_df['y_target'].values))
# print(all_splits[0], all_splits[1])


############## preprocessing
# slice dataframe into time series samples
def _slice_fn(timestamp, set_name):
    '''
    Create a time series sample from a dataframe, starting at the
    given timestamp, with `N_STEPS_SAMPLE_LEN` steps.
    Returns: list of X feature sample + last item ist the Y target value
    '''
    if set_name == 'train': df = train_df
    elif set_name == 'test': df = test_df
    else: raise IOError(f'Unkown datas set {set_name}')
    # slice sample to 24 hour step length
    slice_df = df.loc[df.index > timestamp, :]
    sa_df = slice_df.iloc[:N_STEPS_SAMPLE_LEN, :]
    return sa_df.y.values.tolist() +[float(sa_df.y_target[-1])]


train_tup = train_df.iloc[:train_df.shape[0]-N_STEPS_SAMPLE_LEN].index.values
test_tup = test_df.iloc[:test_df.shape[0]-N_STEPS_SAMPLE_LEN].index.values

with mp.Pool(os.cpu_count()) as p:
    sample_train_lst = p.starmap(_slice_fn, zip(train_tup, itertools.repeat('train')))
    sample_test_lst = p.starmap(_slice_fn, zip(test_tup, itertools.repeat('test')))

n_train_samples = len(sample_train_lst)
print('# Train:', n_train_samples)
n_test_samples = len(sample_test_lst)
print('# Test:', n_test_samples)

# reshape and split features and targets
XY_train = np.array(sample_train_lst).reshape((n_train_samples, N_STEPS_SAMPLE_LEN+1))
XY_test = np.array(sample_test_lst).reshape((n_test_samples, N_STEPS_SAMPLE_LEN+1))

X_train = np.array([sa[:-1] for sa in XY_train])
Y_train = np.array([sa[-1] for sa in XY_train])
X_test = np.array([sa[:-1] for sa in XY_test])
Y_test = np.array([sa[-1] for sa in XY_test])


# # NOTE data augmentation
# from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
# my_aug = (
#     TimeWarp() * 5 
#     + Crop(size=300)
#     + Quantize(n_levels=[10, 20, 30])
#     + Drift(max_drift=(0.1, 0.5)) @ 0.8 
#     + Reverse() @ 0.5 
# )
# X_aug, Y_aug = my_aug.augment(np.array(clean_df['y']), np.array(clean_df['y_target']))


# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


######## XGBoost model
xgb_clf = XGBRegressor()
xgb_clf.fit(X_train, Y_train)


###############################
# Evaluation
y_pred = xgb_clf.predict(X_test)
# MAE
mae = mean_absolute_error(Y_test, y_pred)
print(f"MAE: {mae:.2f}")
# RMSE
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

##### plot predictions (y_pred) over true data (y_true)
eval_df = pd.DataFrame.from_dict({'y_true':Y_test, 'y_pred':y_pred, 'day':test_df.index[N_STEPS_SAMPLE_LEN:]})
ax = eval_df.plot(x='day', y=['y_true', 'y_pred'], kind='line', figsize=(16, 6))
fig = ax.get_figure()
fig.savefig(f'./plots/16_xgboost_cross_validation_{N_STEPS_PRED_HORIZON}.png')