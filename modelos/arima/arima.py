import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import itertools
import os

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

import pmdarima as pm

from utils import show_result_model

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.style.use('fivethirtyeight')
pd.options.display.float_format = '{:.2f}'.format

#TICKER_FILE = "../../dataset/df_itub4_cleansing.csv"
TICKER_FILE = "../../dataset/df_abev3_cleansing.csv"
#TICKER_FILE = "../../dataset/df_vale3_cleansing.csv"

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TICKER_FILE)

df_vale3 = pd.read_csv(
    csv_path,
    encoding='utf8',
    delimiter=',',
    parse_dates=True,
    index_col=0,
    verbose=True
)

n_total = len(df_vale3)
size_train = int(n_total * 0.8)
size_test  = n_total - size_train

df_train = df_vale3['preco'].iloc[:size_train].reset_index(drop=True)
df_test  = df_vale3['preco'].iloc[size_train:size_train + size_test].reset_index(drop=True)

df_train.index = pd.RangeIndex(start=0, stop=len(df_train), step=1)
df_test.index  = pd.RangeIndex(start=size_train, stop=size_train + size_test, step=1)

dict_results = {}

p = q = range(0, 3)
d = range(0, 3)
list_pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in itertools.product(p, d, q)]

def search_best_params_arima_model(df_train, pdq):
    best_model = np.inf
    best_params = (0, 0, 0)
    for param in pdq:
        try:
            model = ARIMA(df_train, order=param)
            results = model.fit()
            print(f'pdq = {param} | AIC = {results.aic}')
            if results.aic < best_model:
                best_model = results.aic
                best_params = param
        except Exception:
            continue
    print(f'best ARIMA: {best_params} | AIC:{best_model}')
    return [best_params, best_model]

list_order_arima = search_best_params_arima_model(df_train=df_train, pdq=list_pdq)
print(list_order_arima)

autoarima_model = pm.auto_arima(
    df_train,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    information_criterion='aic',
    start_p=2,
    start_d=1,
    start_q=2,
    lags=size_test,      
    seasonal=False,
    trace=True
)
print(autoarima_model.order)
print(autoarima_model.aic())

autoarima_model_fit = autoarima_model.fit(y=df_train)
y_forecast = autoarima_model_fit.predict(n_periods=size_test)

_ = show_result_model(
    df_test=df_test,
    y_forecast=y_forecast,
    model_name='arima_model',
    dict_results=dict_results
)

y_true = np.asarray(df_test.values, dtype=float)
y_pred = np.asarray(y_forecast, dtype=float)
y_train = np.asarray(df_train.values, dtype=float)

rmse_value = np.sqrt(np.mean((y_true - y_pred) ** 2))

if y_train.size < 2:
    mase_value = np.inf
else:
    denom = np.mean(np.abs(np.diff(y_train)))
    if denom == 0 or np.isnan(denom):
        mase_value = np.inf
    else:
        mae = np.mean(np.abs(y_true - y_pred))
        mase_value = mae / denom

metrics = {'RMSE': rmse_value, 'MASE': mase_value}
if 'arima_model' in dict_results:
    if isinstance(dict_results['arima_model'], dict):
        dict_results['arima_model'].update(metrics)
    elif isinstance(dict_results['arima_model'], list):
        dict_results['arima_model'].append(metrics)
    else:
        dict_results['arima_model'] = metrics
else:
    dict_results['arima_model'] = metrics

print("dict_results:", dict_results)

x_test = df_test.index.values
y_test = df_test.values
y_hat  = np.asarray(y_forecast)

mse_value  = np.mean((y_test - y_hat) ** 2)
mape_value = np.mean(np.abs((y_test - y_hat) / y_test)) * 100 

title_line1 = f"MAPE = {mape_value:.2f} % | MSE = {mse_value:.2f}"
title_line2 = f"RMSE = {rmse_value:.2f} | MASE = {mase_value:.2f}"

plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test, label='preco')
plt.plot(x_test, y_hat,  label='previsao', linewidth=1.5)

plt.suptitle(f"{title_line1}\n{title_line2}", fontsize=16, y=0.98)
plt.title("")  

plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.93]) 
plt.show()
