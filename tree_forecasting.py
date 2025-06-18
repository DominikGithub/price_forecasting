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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb


###############################################
# Adjust experiment parameters
###############################################

# add artificial time and leg features
DO_EXTEND_FEATURES = False

N_STEPS_PRED_HORIZON = 24
N_STEPS_SAMPLE_LEN = 24

###############################################

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
clean_df['y_target'] = clean_df.y.shift(-N_STEPS_PRED_HORIZON, freq="h")
# fill empty target values
clean_df.loc[clean_df['y_target'].isna(), 'y_target'] = clean_df.loc[clean_df['y_target'].isna(), 'y']

# # plot raw data and targets
# plot_df = clean_df.reset_index()
# ax = plot_df.plot(x='ds', y=['y', 'y_target'], kind='line', figsize=(16, 6))
# fig = ax.get_figure()
# fig.savefig(f'./test_target.png')


### create artifical features
if DO_EXTEND_FEATURES:
    clean_df = clean_df.reset_index()
    # add time features
    clean_df['hour'] = clean_df['ds'].dt.hour
    clean_df['dayofweek'] = clean_df['ds'].dt.dayofweek
    # clean_df['is_weekend'] = clean_df['dayofweek'].isin([5, 6]).astype(int)
    # clean_df['sin_hour'] = np.sin(2 * np.pi * clean_df['hour'] / 24)
    # clean_df['cos_hour'] = np.cos(2 * np.pi * clean_df['hour'] / 24)
    # add leg and rolling features
    clean_df['lag1'] = clean_df['y'].shift(1)
    # clean_df['lag24'] = clean_df['y'].shift(24)
    # clean_df['rolling_mean_24'] = clean_df['y'].rolling(24).mean()
    # clean_df['rolling_std_24'] = clean_df['y'].rolling(24).std()
    clean_df['diff1'] = clean_df['y'].diff(1)                           # difference to previous hour
    # clean_df['diff24'] = clean_df['y'].diff(24)                         # difference to previous day
    # reset index and fill empty (NOTE improve by copy Y to nan's instead of zero)
    clean_df = clean_df.fillna(0)
    clean_df = clean_df.set_index('ds')

    # plot features
    plot_df = clean_df.reset_index().drop(columns=['y', 'y_target'])
    ax = plot_df.plot(x='ds', kind='line', figsize=(16, 6))
    fig = ax.get_figure()
    fig.savefig(f'./plots/21_features.png')

n_features = clean_df.shape[1]-1 # -1 for the target next to feature columns
print('# features:', n_features)


#### cross valdiation split NOTE last month June as test dataset
train_df = clean_df.loc[clean_df.index < '2022-06-01 00:00:00']
test_df = clean_df.loc[clean_df.index >= '2022-06-01 00:00:00']

n_test_samples = test_df.shape[0]
print('# test samples:', n_test_samples)


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
    sa_df = slice_df.iloc[:N_STEPS_SAMPLE_LEN*n_features, :]
    return sa_df.y.values.tolist() +[float(sa_df.y_target[-1])]


train_tup = train_df.iloc[:train_df.shape[0]-N_STEPS_SAMPLE_LEN*n_features].index.values
test_tup = test_df.iloc[:test_df.shape[0]-N_STEPS_SAMPLE_LEN*n_features].index.values

with mp.Pool(os.cpu_count()) as p:
    sample_train_lst = p.starmap(_slice_fn, zip(train_tup, itertools.repeat('train')))
    sample_test_lst = p.starmap(_slice_fn, zip(test_tup, itertools.repeat('test')))

n_train_samples = len(sample_train_lst)
print('# Train:', n_train_samples)
n_test_samples = len(sample_test_lst)
print('# Test:', n_test_samples)

# reshape and split features and targets
XY_train = np.array(sample_train_lst).reshape((n_train_samples, N_STEPS_SAMPLE_LEN*n_features+1))
XY_test = np.array(sample_test_lst).reshape((n_test_samples, N_STEPS_SAMPLE_LEN*n_features+1))

X_train = np.array([sa[:-1] for sa in XY_train])
Y_train = np.array([sa[-1] for sa in XY_train])
X_test = np.array([sa[:-1] for sa in XY_test])
Y_test = np.array([sa[-1] for sa in XY_test])


# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

######## XGBoost model
xgb_clf = XGBRegressor()
xgb_clf.max_depth = 5
xgb_clf.fit(X_train, Y_train)


# #### plot feature importance for trained model
# ax = xgb.plot_importance(xgb_clf, importance_type='gain', max_num_features=15)
# plt.title("Top Feature Importances (Gain)")
# fig = ax.get_figure()
# fig.savefig(f'./feature_importance.png')

# # feature selection
# from sklearn.feature_selection import SelectFromModel
# selector = SelectFromModel(xgb_clf, threshold="mean", prefit=True)
# X_selected = selector.transform(X_train)
# print("Verbliebene Features:", pd.DataFrame(X_train).columns[selector.get_support()])


###############################
## settings
print('Prediction horizon:', N_STEPS_PRED_HORIZON, 'hours')
print('Sample length:', N_STEPS_SAMPLE_LEN, 'hours')
print('Artifical features:', DO_EXTEND_FEATURES)
# print('Model scores:', xgb_clf.get_booster().get_score())
## Evaluation
# predict on test set
y_pred = xgb_clf.predict(X_test)
# MAE
mae = mean_absolute_error(Y_test, y_pred)
print(f"MAE: {mae:.2f}")
# RMSE
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

## Plot predictions (y_pred) over true data (y_true)
eval_df = pd.DataFrame.from_dict({'y_true':Y_test, 'y_pred':y_pred, 'day':test_df.index[N_STEPS_SAMPLE_LEN*n_features:]})
ax = eval_df.plot(x='day', y=['y_true', 'y_pred'], kind='line', figsize=(16, 6))
fig = ax.get_figure()
fig.savefig(f'./plots/16_xgboost_cross_validation_{N_STEPS_PRED_HORIZON}_af{DO_EXTEND_FEATURES}.png')