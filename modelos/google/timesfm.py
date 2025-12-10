import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timesfm

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
pd.options.display.float_format = '{:.2f}'.format

#TICKER_FILE = "../../dataset/df_itub4_cleansing.csv"
TICKER_FILE = "../../dataset/df_abev3_cleansing.csv"
#TICKER_FILE = "../../dataset/df_vale3_cleansing.csv"

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TICKER_FILE)

df = (
    pd.read_csv(csv_path, parse_dates=["data"])
      .sort_values("data")
      .set_index("data")
      .asfreq("B")               
      .ffill()                  
)

n_total = len(df)
n_train = int(n_total * 0.8)
df_contexto = df.iloc[:n_train].copy()     
df_reais    = df.iloc[n_train:].copy()     
horizon     = len(df_reais)                

df_modelo = (
    df_contexto[["preco"]].rename(columns={"preco": "y"})
    .assign(unique_id="ITUB4", ds=lambda x: x.index)
    [["unique_id","ds","y"]]
)

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=1,
        horizon_len=horizon,
        context_len=256,
        num_layers=50
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    )
)

forecast_df = tfm.forecast_on_df(inputs=df_modelo, freq="B", value_name="y")

avail = list(forecast_df.columns)
col = "p50" if "p50" in avail else ("mean" if "mean" in avail else avail[-1])  

y_pred = forecast_df.set_index("ds")[col].reindex(df_reais.index)

assert len(y_pred) == len(df_reais), (len(y_pred), len(df_reais))
assert y_pred.index.equals(df_reais.index)

serie_real = df_reais["preco"].reset_index(drop=True)
serie_prev = pd.Series(y_pred.values, index=serie_real.index)


y_real = serie_real.values
y_pred_vals = serie_prev.values


mape = np.mean(np.abs((y_real - y_pred_vals) / y_real)) * 100
mse = np.mean((y_real - y_pred_vals) ** 2)
rmse = np.sqrt(mse)

train_values = df_contexto["preco"].values
naive_errors = np.abs(train_values[1:] - train_values[:-1])
q = np.mean(naive_errors)
mase = np.mean(np.abs(y_real - y_pred_vals)) / q

plt.figure(figsize=(12, 6))
plt.plot(serie_real.index.values, serie_real.values, label='preco', linewidth=1.8)
plt.plot(serie_prev.index.values, serie_prev.values, label='previsao', linewidth=1.5)
plt.xlabel('Índice (dias de teste)')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)

metric_title = (
    f"MAPE = {mape:.2f} % | MSE = {mse:.2f}\n"
    f"RMSE = {rmse:.2f} | MASE = {mase:.2f}"
)
plt.suptitle(metric_title, fontsize=12, y=0.98)
plt.tight_layout(rect=(0, 0, 1, 0.93))  

plt.show()

print(f"MAPE: {mape:.4f}%")
print(f"MSE : {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MASE: {mase:.6f}")
