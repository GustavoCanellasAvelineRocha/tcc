import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.style.use('fivethirtyeight')
pd.options.display.float_format = '{:.2f}'.format

sys.path.append('../src')
try:
    from utils import (
        path_to_work,
        plot_box_plot,
        save_image,
        save_dataframe,
        test_stationary,
        show_result_model,
    )
except Exception:
    def path_to_work(_): pass
    def show_result_model(df_test, y_forecast, model_name, dict_results):
        plt.figure(figsize=(17,6))
        plt.plot(df_test.index, df_test.values, label='Verdade', alpha=0.6)
        plt.plot(df_test.index, np.array(y_forecast).reshape(-1), label='Previsão', lw=2)
        plt.legend(); plt.title(model_name)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPT_DIR, 'resultados')
os.makedirs(OUTDIR, exist_ok=True)

TICKER_FILE = "../../dataset/df_itub4_cleansing.csv"
#TICKER_FILE = "../../dataset/df_abev3_cleansing.csv"
#TICKER_FILE = "../../dataset/df_vale3_cleansing.csv"

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TICKER_FILE)

df_vale3 = pd.read_csv(
    csv_path, encoding='utf8', delimiter=',',
    parse_dates=True, index_col=0, verbose=True
).sort_index()

expected_cols = ['preco','residuos','tendencia','sazonalidade','diff_1','diff_2','diff_3','diff_4','diff_5']
missing = set(expected_cols) - set(df_vale3.columns)
assert not missing, f"Faltam colunas no cleansing: {missing}"
df_vale3 = df_vale3[expected_cols].copy()

print(df_vale3.info())
print(df_vale3.head())

size_train    = 1960
size_test     = 491

df_train = df_vale3.iloc[:size_train].copy()
df_test  = df_vale3.iloc[size_train:size_train+size_test].copy()

print(f"Total training = {len(df_train)}")
print(f"Total testing  = {len(df_test)}")
print(df_train.index[[0, -1]])
print(df_test.index[[0, -1]])

dict_results = {}

train_max = df_train.max()
train_min = df_train.min()

train = (df_train - train_min) / (train_max - train_min)
test  = (df_test  - train_min) / (train_max - train_min)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs, dtype='float32'), np.array(ys, dtype='float32')

time_steps = 1
X_train, y_train = create_dataset(train, train['preco'], time_steps)
X_test,  y_test  = create_dataset(test,  test['preco'],  time_steps)

print("Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model_lstm = Sequential(name='lstm_vale3')
model_lstm.add(LSTM(units=len(df_train.columns),
                    return_sequences=True,
                    input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=10, return_sequences=True)); model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=10, return_sequences=True)); model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=10));                         model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mape'])
model_lstm.summary()

history = model_lstm.fit(
    X_train, y_train,
    epochs=1000, batch_size=30,
    shuffle=False, validation_split=0.30, verbose=0
)

print(history.history.keys())
best_epochs = history.history["loss"].index(min(history.history["loss"]))
print("best_epochs idx:", best_epochs)
print("min_loss:", min(history.history["loss"]))

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title('loss'); plt.legend(); plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title('Loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'lstm_loss.png'), dpi=300)
plt.close()

y_pred = model_lstm.predict(X_test, verbose=0).reshape(-1)

y_test_den  = y_test * (train_max[0] - train_min[0]) + train_min[0]
y_pred_den  = y_pred * (train_max[0] - train_min[0]) + train_min[0]
y_train_den = y_train * (train_max[0] - train_min[0]) + train_min[0]

print("Len y_train_den:", len(y_train_den))
print("Len y_test_den :", len(y_test_den))

N = min(len(df_test), len(y_pred_den))

y_true  = df_test['preco'][:N].to_numpy(dtype=float)
y_predo = y_pred_den[:N].astype(float)
y_train_lvl = df_train['preco'].to_numpy(dtype=float)

mse_value  = float(np.mean((y_true - y_predo) ** 2))
mape_value = float(np.mean(np.abs((y_true - y_predo) / y_true)) * 100.0)
rmse_value = float(np.sqrt(mse_value))

if y_train_lvl.size < 2:
    mase_value = np.inf
else:
    denom = float(np.mean(np.abs(np.diff(y_train_lvl))))
    if denom == 0 or np.isnan(denom):
        mase_value = np.inf
    else:
        mae = float(np.mean(np.abs(y_true - y_predo)))
        mase_value = float(mae / denom)

bias_value = float(np.mean(y_predo - y_true))

metrics = {
    'MAPE(%)': mape_value,
    'MSE': mse_value,
    'RMSE': rmse_value,
    'MASE': mase_value,
    'Bias(ME)': bias_value
}
dict_results['model_lstm'] = metrics
print("dict_results:", dict_results)

title_line1 = f"MAPE = {mape_value:.2f} % | MSE = {mse_value:.2f}"
title_line2 = f"RMSE = {rmse_value:.2f} | MASE = {mase_value:.2f}"
title_line3 = f"Bias = {bias_value:.2f}"

x_test = df_test.index[:N].values
y_test_plot = df_test['preco'][:N].values
y_hat  = y_predo

plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test_plot, label='Real')
plt.plot(x_test, y_hat,       label='Forecast', linewidth=1.5)

plt.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=16, y=0.97)
plt.title("") 
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(os.path.join(OUTDIR, 'lstm_forecast.png'), dpi=300)
plt.show()
plt.close()

metrics_row = {
    'MAPE(%)': mape_value,
    'MSE': mse_value,
    'RMSE': rmse_value,
    'MASE': mase_value,
    'Bias(ME)': bias_value
}
pd.DataFrame([metrics_row]).to_csv(
    os.path.join(OUTDIR, 'metrics_lstm.csv'),
    index=False, encoding='utf-8'
)

print(f"Arquivos salvos em: {OUTDIR}")
