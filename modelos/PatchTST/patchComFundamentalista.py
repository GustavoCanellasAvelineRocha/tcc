import os, sys, warnings, pathlib, importlib.util, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

sys.path.append('../src')
try:
    from utils import path_to_work
except Exception:
    def path_to_work(_): pass

HERE = pathlib.Path(__file__).resolve().parent
SUPERVISED_ROOT = HERE / "PatchTST" / "PatchTST_supervised"
PATCHTST_FILE = SUPERVISED_ROOT / "models" / "PatchTST.py"

spec = importlib.util.spec_from_file_location("patchtst_mod", str(PATCHTST_FILE))
patchtst_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patchtst_mod)
PatchTST = patchtst_mod.Model

TICKER = "ABEV3" 
FUNDIDO_PATH = HERE / "../../dataset/dados_fundamentalistas/dados_cleansing_fundidos" / f"{TICKER}_cleansing_fundamentalista.csv"

TOTAL_TARGET, TRAIN_LEN, TEST_LEN = 2451, 1960, 491
IS_VALE3 = (TICKER.upper() == "VALE3")

CSV_PATH = FUNDIDO_PATH  

EXPECTED_COLS = [
    'preco','residuos','tendencia','sazonalidade',
    'diff_1','diff_2','diff_3','diff_4','diff_5',
    'P/VP','LL','P/L','EBIT TTM','ROE','EBIT/P'
]

df = pd.read_csv(CSV_PATH, encoding='utf8', delimiter=',')
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df[df["time"].notna()].set_index("time").sort_index()

df[['P/VP','LL','P/L','EBIT TTM','ROE','EBIT/P']] = df[['P/VP','LL','P/L','EBIT TTM','ROE','EBIT/P']].ffill()

df = df[EXPECTED_COLS].tail(TOTAL_TARGET).copy()
df_train, df_test = df.iloc[:TRAIN_LEN].copy(), df.iloc[TRAIN_LEN:].copy()

train_min, train_max = df_train.min(), df_train.max()
train_rng = (train_max - train_min).replace(0, 1.0)
train = (df_train - train_min) / train_rng
test  = (df_test  - train_min) / train_rng

# Janelas para ITUB4 e ABEV3, mudando para VALE3
SEQ_LEN = 96
if IS_VALE3:
    SEQ_LEN = 160

PRED_LEN = 5

def create_dataset(df_norm, target_col='preco', seq_len=21, pred_len=5):
    Xs, ys = [], []
    for i in range(len(df_norm) - seq_len - pred_len + 1):
        Xs.append(df_norm.iloc[i:i+seq_len].values)                             
        ys.append(df_norm[target_col].iloc[i+seq_len:i+seq_len+pred_len].values) 
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(train, 'preco', SEQ_LEN, PRED_LEN)
df_combined_test = pd.concat([train.tail(SEQ_LEN), test], axis=0)
X_test_roll, y_test_roll = create_dataset(df_combined_test, 'preco', SEQ_LEN, PRED_LEN)

class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

val_split = 0.20
n_train = int(len(X_train) * (1 - val_split))

tr_ds, va_ds = DS(X_train[:n_train], y_train[:n_train]), DS(X_train[n_train:], y_train[n_train:])
tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True,  drop_last=False)
va_dl = DataLoader(va_ds, batch_size=512, shuffle=False, drop_last=False)
te_dl = DataLoader(DS(X_test_roll, y_test_roll), batch_size=512, shuffle=False, drop_last=False)

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configs do modelo
def build_configs(seq_len: int, pred_len: int, enc_in: int, c_out: int = 1):
    class Cfg: pass
    cfg = Cfg()
    cfg.seq_len   = seq_len
    cfg.pred_len  = pred_len
    cfg.enc_in    = enc_in
    cfg.c_out     = c_out

    cfg.e_layers  = 4
    cfg.d_model   = 128
    cfg.n_heads   = 8
    cfg.d_ff      = 256
    cfg.activation= 'gelu'
    cfg.pre_norm  = False
    cfg.individual= False
    cfg.head_type = 'prediction'

    cfg.patch_len = max(1, min(16, seq_len))
    cfg.stride    = 1 if seq_len < 16 else 8
    cfg.padding       = 0
    cfg.padding_patch = 'end'

    cfg.revin         = True
    cfg.affine        = True
    cfg.subtract_last = False
    cfg.decomposition = False
    cfg.kernel_size   = 25

    cfg.dropout       = 0.2
    cfg.fc_dropout    = 0.10
    cfg.head_dropout  = 0.05

    cfg.task_name     = 'long_term_forecast'
    cfg.embed         = 'timeF'
    cfg.output_attention = False
    return cfg

cfg = build_configs(SEQ_LEN, PRED_LEN, enc_in=X_train.shape[2], c_out=1)
model = PatchTST(cfg).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Epocas para ITUB4 e ABEV3, mudando para VALE3
EPOCHS, patience = 250, 250
if (TICKER == "VALE3"):
    EPOCHS, patience = 50, 50

total_steps = len(tr_dl) * EPOCHS
warmup_steps = int(0.05 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# treinamento do modelo
best_val, bad, best_state = float('inf'), 0, None
train_hist, val_hist = [], []
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    for xb, yb in tr_dl:
        xb, yb = xb.to(device), yb.to(device)        
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            yhat = model(xb)[:, :, 0]
            loss = loss_fn(yhat, yb)                   # MSE em TODO o horizonte
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        tr_loss += loss.item() * xb.size(0)
        global_step += 1
        scheduler.step()

    tr_loss /= len(tr_dl.dataset)

    # validação
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in va_dl:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)[:, :, 0]
            va_loss += loss_fn(yhat, yb).item() * xb.size(0)
    va_loss /= len(va_dl.dataset)

    train_hist.append(tr_loss)
    val_hist.append(va_loss)

    if va_loss < best_val:
        best_val, bad = va_loss, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
preds = []
with torch.no_grad():
    for xb, _ in te_dl:
        xb = xb.to(device)
        yhat = model(xb)[:, :, 0].cpu().numpy()      
        preds.append(yhat)
        
preds = np.concatenate(preds, axis=0)[:, 0]          
y_true = y_test_roll[:, 0]                         

preco_scale = (train_max['preco'] - train_min['preco'])
preco_min   =  train_min['preco']
y_pred_den  = preds  * preco_scale + preco_min
y_true_den  = y_true * preco_scale + preco_min

truth_index = df_test.index[:len(y_pred_den)]
y_true_series = pd.Series(y_true_den.reshape(-1), index=truth_index, name='preco')
y_pred_series = pd.Series(y_pred_den.reshape(-1), index=truth_index, name='previsao')

mse  = float(mean_squared_error(y_true_series, y_pred_series))
rmse = float(np.sqrt(mse)) 
mape = float(mean_absolute_percentage_error(y_true_series, y_pred_series) * 100.0)

train_preco = df_train['preco'].astype(float).values
naive_diffs = np.abs(train_preco[1:] - train_preco[:-1])
naive_scale = float(naive_diffs.mean()) if len(naive_diffs) > 0 else np.nan

mae_model = float(np.mean(np.abs(y_true_den - y_pred_den)))
mase = float(mae_model / naive_scale) if naive_scale not in (0.0, np.nan) else np.nan

CSV_NAME   = pathlib.Path(CSV_PATH).stem
TICKER_TAG = ''.join(ch if ch.isalnum() else '_' for ch in CSV_NAME)
OUTDIR     = (HERE / 'resultadosFundamentalista' / TICKER_TAG)
OUTDIR.mkdir(parents=True, exist_ok=True)

def smooth(x, k=7):
    if len(x) < k: return x
    w = np.ones(k)/k
    return np.convolve(np.array(x, dtype=float), w, mode='same')

fig1_path = OUTDIR / f'learning_curves_{TICKER_TAG}.png'
plt.figure(figsize=(10,4))
plt.plot(smooth(train_hist,7), label="Training Loss")
plt.plot(smooth(val_hist,7),   label="Validation Loss")
plt.title("Convergência da taxa de perda (loss)")
plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
plt.legend(loc='best'); plt.tight_layout()
plt.savefig(fig1_path, dpi=300); plt.close()

fig2_path = OUTDIR / f'patchtst_pred_{TICKER_TAG}.png'
fig, ax = plt.subplots(figsize=(16,10))
ax.plot(y_true_series.index, y_true_series.values, label='preco',      linewidth=1.6, alpha=0.95)
ax.plot(y_pred_series.index, y_pred_series.values, label='previsao', linewidth=1.6, alpha=0.95)
ax.set_title(
    f"PREVISÃO MAPE = {mape:.2f}% | MSE = {mse:.2f} | RMSE = {rmse:.2f} | MASE = {mase:.3f}",
    fontsize=22, pad=16
)
ax.set_xlabel("Tempo (dias)", fontsize=14); ax.set_ylabel("Preço (R$)", fontsize=14)
ax.grid(True, which='major', alpha=0.25); ax.legend(loc='upper right', frameon=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')); fig.autofmt_xdate()
plt.savefig(fig2_path, dpi=300); plt.close()

csv_pred_path = OUTDIR / f'predicoes_pred_{TICKER_TAG}.csv'
pd.concat([y_true_series.rename('preco_real'),
           y_pred_series.rename('preco_previsto')], axis=1).to_csv(csv_pred_path, encoding='utf-8')

with open(OUTDIR / 'metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"MAPE: {mape:.4f}%\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")      
    f.write(f"MASE: {mase:.6f}\n")      
    f.write(f"Best Val Loss: {best_val:.6f}\n")

print(f"\nPredição salva em: {OUTDIR}")
print(f"MAPE={mape:.4f}% | MSE={mse:.6f} | RMSE={rmse:.6f} | MASE={mase:.6f}")