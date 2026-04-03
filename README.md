# Energy Price Forecasting 
![CatBoost](https://img.shields.io/badge/CatBoost-yellow) 
![XGBoost](https://img.shields.io/badge/XGBoost-yellow) 
![LightGBM](https://img.shields.io/badge/LightGBM-yellow) 
![Prophet](https://img.shields.io/badge/Prophet-yellow) 
![Python](https://img.shields.io/badge/Python–3.12-blue)

Time series public market auction price forecasting. Comparing different prediction horizon scenarios (hourly and daily), prediction models and contribution of feature engineering.

#### Results overview

|   Model  | Art. Feat |  Period [h] |   Horizon [h] |    MAE   |   RMSE   |
| -------- | --------- | ----------- | ------------- | -------- | -------- |
|  Prophet |    No     |      1      |       1       |   47.66  |   61.33  |
|  Prophet |    No     |      24     |       24      |   50.09  |   64.45  |
|  XGBoost |    No     |      1      |       1       |   21.08  |   29.37  |
|  XGBoost |    No     |      6      |       1       |   16.82  |   24.28  |
|  XGBoost |    No     |      24     |       1       |   15.27  |   21.01  |
|  XGBoost |    No     |      1      |       24      |   43.15  |   59.91  |
|  XGBoost |    No     |      6      |       24      |   46.72  |   64.49  |
|  XGBoost |    No     |      24     |       24      |   46.24  |   62.84  |
|  XGBoost |    Yes    |      1      |       1       |   17.21  |   25.54  |
|  XGBoost |    Yes    |      6      |       1       |   14.87  |   20.81  |
|  XGBoost |    Yes    |      24     |       1       |   16.65  |   23.35  |
|  XGBoost |    Yes    |      1      |       24      |   46.32  |   63.47  |
|  XGBoost |    Yes    |      6      |       24      |   47.19  |   64.21  |
|  XGBoost |    Yes    |      24     |       24      |   49.48  |   65.98  |
| CatBoost |    No     |      1      |       1       |   20.88  |   29.26  |
| CatBoost |    No     |      6      |       1       |   15.84  |   23.37  |
| CatBoost |    No     |      24     |       1       |   14.58  |   20.72  |
| CatBoost |    No     |      1      |       24      |   42.30  |   58.95  |
| CatBoost |    No     |      6      |       24      |   43.43  |   61.04  |
| CatBoost |    No     |      24     |       24      |   44.10  |   59.13  |
| CatBoost |    Yes    |      1      |       1       |   16.12  |   23.80  |
| CatBoost |    Yes    |      6      |       1       |   ![14.40](https://img.shields.io/badge/14.40-brightgreen)  |  ![20.19](https://img.shields.io/badge/20.19-brightgreen) |
| CatBoost |    Yes    |      24     |       1       |   15.08  |   21.46  |
| CatBoost |    Yes    |      1      |       24      |   43.16  |   60.62  |
| CatBoost |    Yes    |      6      |       24      |   44.40  |   59.07  |
| CatBoost |    Yes    |      24     |       24      |   ![42.03](https://img.shields.io/badge/42.03-purple)  |  ![58.99](https://img.shields.io/badge/58.99-purple)  |
| LightGBM |    No     |      6      |       1       |   16.52  |   25.62  |
| LightGBM |    No     |      24     |       24      |   43.47  |   59.39  |
| LightGBM |    Yes    |      6      |       1       |   17.01  |   26.13  |
| LightGBM |    Yes    |      24     |       24      |   44.79  |   62.75  |

### Quick start 
```
git clone ...
cd price_forecasting
poetry install
poetry shell
(venv) python prophet_forecasting.py                            # evaluates Prophet model
(venv) python tree_forecasting.py -m catboost -l 24 -z 24       # evaluates tree based models (catboost, xgboost, lightgbm)
(venv) python calculate_profit.py                               # calculate trading profit for all models
(venv) python optimization.py                                   # evaluates optimization strategy
```


## Dataset 
Hourly electicity prices from public european energy market over a 6 month period time window.

Data sources: 
- [https://www.smard.de](https://www.smard.de) (Bundesnetzagentur) 
- [https://transparency.entsoe.eu](https://transparency.entsoe.eu) 

Download as .csv file in various time resolutions.

## EDA 

### Table annotations 
Available dataset columns names.

- MTU (CET/CEST) $\rightarrow$ Time intervals [FROM, TO] in UTC+1 timezone 
- Day-ahead Price [EUR/MWh] $\rightarrow$ Target price column to be predicted 
- Currency $\rightarrow$ Price unit Euro (irrelevant for modeling) 
- BZN|DE-LU $\rightarrow$ Bidding zone Germany/Luxembourg 

---

### Market price raw data 
Electricity price of the public auction market.

![Raw pricing data](./plots/0_raw_prices.png)
_Y Price [€/MWh] & ds [hour]_

__Observations__ 
- Noisy short term variability, next to long term repetitive cyclic patterns 
- External factors seems to cause unusual large outlier in early March 


### Time series statistics
![Raw data statistics](./plots/1_eda_stats.png)

_ds: Time | y: Price_


### Preprocessing daily sample distribution 

![hourly sampling histogram](./plots/2_histogram_hourly_distribution.png)
_(Log) Price samples per day historgram_

Gap localization over time.
![data gaps](./plots/3_gap_line.png)
_Samples count per day_

Data cleansing by interpolation over small time step gaps.

#### Feature engineering 
Two situations with regards to the feature dimensions have been compared for the tree models. Raw time series only as feature input, in contrast to extracting additional common metrics from the price.

![Raw pricing data](./plots/21_features.png)
_Features over time_


## Forecast model (1. Prophet) 
__POC parametrization__ 
- Initial window size in days (120) $\rightarrow$ training window (the bigger the better) 
- Horizon in hours $\rightarrow$ prediction step size (how far to predict into the future) 
- Period in hours $\rightarrow$ number of prediction steps 
![predictions](./plots/5_predictions.png) 
_Y [Price in €/MWh] over time_

Legend: 
- Historical observations (black dots) 
- Confidence interval (light blue band) 
- Predictions (blue dense line) 
- Upper threshold limit (dashed black line) 

### Test set evaluation 
Time series cross validation is used to measure the forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. The forecasted values (_yhat_) are compared to the actual (_y_) values.

![Test set prohpet](./plots/6_cross_validation.png) 
_Prediction & observations over time for daily forecast horizon [Price in €/MWh]_

Metrics are below the pricings standard deviation of 90.656821, which means they are reasonable, but error metrics are still at quite high level. Hence the model did derive valuable information from the data, but it can be assumed that there is quite some potential left with dataset preprocessing and model selection. And most importantly the models parameters (e.g. sampling strategy) are just chosen for quick experimentation but not for optimal results and need more adjustment.


#### Review 

Prophet model for time series forecasting is exhausting its capability to handle high frequency, volatile from external unkown factors in the data and struggles to model irregular patterns.


## Forecast model (2. XGBoost) 
Boosting (unbalanced) tree based regression model for time series forecasting. 

### Test set evaluation  
Test set size of 696 June samples (deducting samples from gap between cross validation splits).
Two scenarios of hourly and one day ahead forecast horizons.

![test set prediction xgboost](./plots/16_xgboost_cross_validation_24_afFalse.png) 
_Prediction & observations for daily forecast horizon, no artifical features [Price in €/MWh]_

![test set prediction xgboost](./plots/16_xgboost_cross_validation_1_afFalse.png) 
_Prediction & observations for hourly forecast horizon, no artifical features [Price in €/MWh]_

__XGBoost with extended input feature dimensions__
![test set prediction xgboost](./plots/16_xgboost_cross_validation_24_afTrue.png) 
_Prediction & observations for daily forecast horizon, artifical features added [Price in €/MWh]_

![test set prediction xgboost](./plots/16_xgboost_cross_validation_1_afTrue.png) 
_Prediction & observations for hourly forecast horizon, artifical features added [Price in €/MWh]_


#### Review 
The XGBoost model achieves significantly smaller error metrics on the evaluation sets for the short term prediction scenario and handles non saisonale spiky patterns better. While the tree based model outperforms the prohpet model on long term predictions on a smaller distance. Data set extension adding artifical feature dimensions, does only contribute benefitial to the prediction accuracy in the short horizon scenario.


## Forecast model (3. CatBoost) 
CatBoost is known to excel even without extensive hyperparameter optimization building symmetric trees. 

### Test set evaluation  
Test set size of 696 June samples (deducting samples from gap between cross validation splits).
Two scenarios of hourly and one day ahead forecast horizons.

![test set prediction catboost](./plots/16_catboost_cross_validation_24_afFalse.png) 
_Prediction & observations for daily forecast horizon, no artifical features [Price in €/MWh]_

![test set prediction catboost](./plots/16_catboost_cross_validation_1_afFalse.png) 
_Prediction & observations for hourly forecast horizon, no artifical features [Price in €/MWh]_

__CatBoost with extended input feature dimensions__
![test set prediction catboost](./plots/16_catboost_cross_validation_24_afTrue.png) 
_Prediction & observations for daily forecast horizon, artifical features added [Price in €/MWh]_

![test set prediction catboost](./plots/16_catboost_cross_validation_1_afTrue.png) 
_Prediction & observations for hourly forecast horizon, artifical features added [Price in €/MWh]_

#### Review 
Catboost outperforms XGBoost slightly and Prophet forecasting scenario. In contrast to XGboost it benefits from extended feature dimension even in small dataset challenges.

Overall it can be observed that the fitting effort for tree based models are multiple magnitudes smaller, than for non tree-based additive regression models like Prohpet. While both tree model approaches also achieved better results on the test set for energy price forecasting. 


## Optimization Strategy

__Task:__ Buy energy at cheap prices and store in batteries, to sell at future higher prices.

### Problem formalization 

Charging speed at time t: $c_t$

__Battery constraints__ 

Total  capacity: $0 <= SOC_t <= 1MWh$

Charging speed: $-1 <= c_t <= 1MWh$


__Trading actions__ 

$`c_t > 0 → `$ Charging/Buy 

$`c_t = 0 → `$ Idle/Hold 

$`c_t < 0 → `$ Discharging/Sell 


__State update__ $`SOC_{t+1} = SOC_t + c_t`$

__Optimizable cost function__ $`max ∑_t = price_t * c_t`$

### Optimization Results

Using Linear Programming, we calculate profit by buying low (charging) and selling high (discharging).

| Model | Total Profit | Avg Profit | vs Perfect |
|-------|-------------|------------|-----------|
| Perfect foresight | 5,894.65 € | 9.82 €/MWh | - |
| XGBoost (24h) | 9,702.07 € | 16.17 €/MWh | +65% |
| LightGBM (24h) | 7,867.31 € | 13.11 €/MWh | +33% |
| CatBoost (24h) | 6,822.73 € | 11.37 €/MWh | +16% |
| Prophet | 4,277.81 € | 7.13 €/MWh | -27% |

**Why XGBoost wins despite higher MAE:**

| Model | Price Std | Min | Max | Range |
|-------|-----------|-----|-----|-------|
| Actual | 76.2 | -0.6 | 500.0 | 500.6 |
| XGBoost | 91.8 | 6.6 | **553.9** | 547.3 |
| LightGBM | 89.3 | 18.2 | 502.1 | 483.9 |
| CatBoost | 87.5 | 5.8 | 438.0 | 432.2 |
| Prophet | 49.1 | 38.0 | 280.5 | 242.5 |

- **XGBoost captures extremes** - max 553.9 (overshoots actual) = stronger sell signals
- **CatBoost is conservative** - max ~440 = weaker signals, lower profit
- **MAE ≠ profit-optimal** - being off on extremes hurts MAE but helps profit
- **For arbitrage, ordering matters** - identifying cheap vs expensive hours > exact values

### Model MAE Comparison

| Model | Horizon | MAE | RMSE |
|-------|---------|-----|------|
| CatBoost | 1h | 14.40 | 20.19 |
| CatBoost | 24h | 42.03 | 58.99 |
| LightGBM | 24h | 43.47 | 59.39 |
| XGBoost | 24h | 46.24 | 62.84 |

**Key findings:**
- CatBoost has lowest MAE but XGBoost has highest profit
- For arbitrage, prediction variance matters more than point accuracy
- Feature engineering improves MAE by ~5% but doesn't improve profit significantly

