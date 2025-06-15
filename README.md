# Energy price forecasting 

Time series day ahead public market auction price forecasting.

### Dataset 
Hourly electicity prices from public european energy market over a 6 month period time window.

Data sources: 
- [https://www.smard.de](https://www.smard.de) (Bundesnetzagentur) 
- [https://transparency.entsoe.eu](https://transparency.entsoe.eu) 

Download as .csv file in various time resolutions.

## EDA 

### Tabular data annotation overview 
Available dataset columns are named are as follows.

- MTU (CET/CEST) -> Time intervals [FROM, TO] in UTC+1 timezone 
- Day-ahead Price [EUR/MWh] -> Target price column to be predicted 
- Currency -> Price unit Euro (while constant, assumed to be irrelevant for modeling) 
- BZN|DE-LU -> Bidding zone Germany/Luxembourg, (irrelevant for modeling) 

---

### Raw time series - visualization
![Raw pricing data](./plots/0_raw_prices.png)
<center>Y [Price in €/MWh]</center>

- Assumend density pattern change in March in the raw data 
- Large outlier in early March 


### Raw time series - statistics
![Raw data statistics](./plots/1_eda_stats.png)
<center>ds: Time | y: price</center>


### Sampling distribution

![hourly sampling histogram](./plots/2_histogram_hourly_distribution.png)
<center>(Log) Price samples per day historgram</center>

![data gaps](./plots/3_gap_line.png)
<center>Samples count per day</center>

Data verification after data cleansing, preprocessing and interpolation. 
![hourly distribution](./plots/4_histogram_hours_per_day.png)
<center>Price samples per day historgram</center>

## Forecast model
__POC parametrization__ 
- Initial window size: 120 days $\rightarrow$ training window (the bigger the better) 
- Horizon: 1 hours $\rightarrow$ prediction step size (how far to predict into the future) 
- Period: 1 hours $\rightarrow$ number of prediction steps (how often to make predictions) 
![predictions](./plots/5_predictions.png) 
<center>Vertical axis: Y [Price in €/MWh]</center> 

Legend: 
- Historical observations (black dots) 
- Confidence interval (light blue band) 
- Predictions (blue dense line) 
- Upper threshold limit (dashed black line) 

TODO adjust model parameters for improved results and day ahead predictions.

### Cross validation
Time series cross validation is used to measure the forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point.  The forecasted values are compared to the actual values.

![Cross validation](./plots/6_cross_validation.png) 
<center>Prediction & observations over time [Price in €/MWh]</center>

Legend: 
- _y_: Ground truth
- _yhat_: prediction


__Evaluation metrics:__
| Experiment   | Pred. Period | Pred. Horizon |      MAE     |     RMSE     |
| ------------ | ------------ | ------------- | ------------ | ------------ |
|  Next hour   |       1      |       1       |     47.66    |      61.33   |
|  Day ahead   |       24     |       24      |     50.09    |      64.45   |


Metrics are below the pricings standard deviation of 90.656821, which means they are reasonable, but at the same time the difference is small. Hence the model did derive valuable information from the data, but it can be assumed that there is quite some potential left with dataset preprocessing and model selection. And most importantly the models parameters are just chosen for quick experimentation but not for optimal results for far and need more adjustment.


## Optimization Approach

__Task:__ Buy energy cheap and store in batteries to sell at high prices.

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



<!-- ## Approach comparison

Historic prices profit: 21616.323853726

Predicted prices profit: <TODO fix model and recalculate> 

Prediction vs historic price strategy difference: <TODO show difference> -->

