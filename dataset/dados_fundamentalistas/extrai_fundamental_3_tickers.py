import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).parent.resolve()
CONVERTIDO_DIR = BASE_DIR / "convertido"
# CSVs estão na raiz /dataset/
PRECO_DIR = BASE_DIR.parent
OUT_DIR = BASE_DIR / "dados_fundamentalistas_filtrados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 1000.0  # relatórios em milhares

# Ações em circulação
SHARES = {
    "VALE3": 4_590_000_000.0,
    "ABEV3": 15_742_213_579.0,
    "ITUB4": 5_330_429_488.0,
}

# Nomes exatos nos teus arquivos
PL_NAME = "Patrimônio Líquido"
LL_NAME = "Lucro/Prejuízo do Período"

EBIT_COMPONENTS = [
    "Resultado Bruto",
    "Despesas Com Vendas",
    "Despesas Gerais e Administrativas",
    "Perdas pela Não Recuperabilidade de Ativos\xa0",  # NBSP
    "Outras Receitas Operacionais",
    "Outras Despesas Operacionais",
    "Resultado da Equivalência Patrimonial",
]

EXPECTED_FILES = {
    "VALE3": ("balanco_vale3__Bal._Patrim..xlsx", "balanco_vale3__Dem._Result..xlsx"),
    "ABEV3": ("balanco_abev3__Bal._Patrim..xlsx", "balanco_abev3__Dem._Result..xlsx"),
    "ITUB4": ("balanco_itub4__Bal._Patrim..xlsx", "balanco_itub4__Dem._Result..xlsx"),
}

# =========================
# Funções utilitárias
# =========================
def read_excel_fixed(path: Path) -> pd.DataFrame:
    """Lê no formato dos teus arquivos: col0=Conta; col1..=datas -> transpõe."""
    try:
        df = pd.read_excel(path, sheet_name=0, header=1)
        if not isinstance(df.columns[0], str):
            df = pd.read_excel(path, sheet_name=0, header=0)
    except Exception:
        df = pd.read_excel(path, sheet_name=0, header=0)
    df = df.rename(columns={df.columns[0]: "Conta"}).set_index("Conta").T
    idx = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
    df = df[idx.notna()]
    df.index = idx[idx.notna()]
    return df.apply(pd.to_numeric, errors="coerce")

def compute_ebit_q_industrial(dre_df: pd.DataFrame) -> pd.DataFrame:
    """EBIT trimestral (VALE3/ABEV3) via componentes exatos."""
    sub = dre_df[EBIT_COMPONENTS].copy() * SCALE_FACTOR
    ebit_q = (
        sub["Resultado Bruto"]
        - sub["Despesas Com Vendas"]
        - sub["Despesas Gerais e Administrativas"]
        + sub["Perdas pela Não Recuperabilidade de Ativos\xa0"]
        + sub["Outras Receitas Operacionais"]
        + sub["Outras Despesas Operacionais"]
        + sub["Resultado da Equivalência Patrimonial"]
    )
    return pd.DataFrame({"EBIT_Q": ebit_q})

# =========================
# Pipeline por ticker
# =========================
def compute_daily_fundamentals(ticker: str) -> pd.DataFrame:
    bal_xlsx, dre_xlsx = EXPECTED_FILES[ticker]
    bal_path = CONVERTIDO_DIR / bal_xlsx
    dre_path = CONVERTIDO_DIR / dre_xlsx

    df_bal = read_excel_fixed(bal_path)
    df_dre = read_excel_fixed(dre_path)

    df_pl = df_bal[[PL_NAME]].rename(columns={PL_NAME: "PL_Total"}) * SCALE_FACTOR
    df_ll = df_dre[[LL_NAME]].rename(columns={LL_NAME: "LL_Q"}) * SCALE_FACTOR

    # EBIT: ABEV3 e VALE3 seguem industrial; ITUB4 é bancário
    if ticker in ("VALE3", "ABEV3"):
        df_ebit_q = compute_ebit_q_industrial(df_dre)
    else:  # ITUB4
        df_ebit_q = df_dre[["Resultado Operacional"]].rename(columns={"Resultado Operacional": "EBIT_Q"}) * SCALE_FACTOR

    df_q = df_pl.join(df_ll, how="outer").join(df_ebit_q, how="outer")

    # TTM
    df_ttm = df_q.copy()
    df_ttm["LL_TTM"] = df_ttm["LL_Q"].rolling(window=4, min_periods=4).sum()
    df_ttm["EBIT_TTM"] = df_ttm["EBIT_Q"].rolling(window=4, min_periods=4).sum()

    # Por ação
    shares = SHARES[ticker]
    df_ttm["VPA"] = df_ttm["PL_Total"] / shares
    df_ttm["LPA_TTM"] = df_ttm["LL_TTM"] / shares
    df_ttm["EBIT_A"] = df_ttm["EBIT_TTM"] / shares

    # --- Merge com preços (colunas em português) ---
    preco_path = PRECO_DIR / f"{ticker}_dados_b3.csv"
    if not preco_path.exists():
        raise FileNotFoundError(f"Arquivo de preços não encontrado: {preco_path}")
    print(f"[INFO] Lendo preços de: {preco_path}")

    df_stock = pd.read_csv(preco_path, sep=",")
    df_stock["data"] = pd.to_datetime(df_stock["data"])
    df_stock = df_stock.rename(columns={"fechamento": "Close"})
    df_stock = df_stock.set_index("data")[["Close"]]

    # Junção e preenchimento
    cols_ff = ["PL_Total", "LL_TTM", "EBIT_TTM", "VPA", "LPA_TTM", "EBIT_A"]
    df = df_stock.join(df_ttm[cols_ff]).sort_index()
    df[cols_ff] = df[cols_ff].ffill()
    df = df.dropna()

    # Métricas
    df["P/VP"] = df["Close"] / df["VPA"]
    df["P/L"] = np.where(df["LPA_TTM"].abs() > 1e-9, df["Close"] / df["LPA_TTM"], np.nan)
    df["ROE"] = np.where(df["PL_Total"].abs() > 1e-9, df["LL_TTM"] / df["PL_Total"], np.nan)
    df["EBIT/P"] = np.where(df["Close"].abs() > 1e-9, df["EBIT_A"] / df["Close"], np.nan)

    # Saída final
    df_final = df.rename(columns={
        "LL_TTM": "LL",
        "EBIT_TTM": "EBIT TTM",
    })
    df_final_output = df_final[["P/VP", "LL", "P/L", "EBIT TTM", "ROE", "EBIT/P"]]
    return df_final_output

# =========================
# MAIN
# =========================
def main():
    tickers = ["VALE3", "ABEV3", "ITUB4"]
    for t in tickers:
        print(f"[INFO] Processando {t} ...")
        df_out = compute_daily_fundamentals(t)
        out_csv = OUT_DIR / f"{t}_dados_fundamentalistas_diarios_final.csv"
        df_out.to_csv(out_csv, date_format="%Y-%m-%d")
        print(f"[OK] Salvo: {out_csv} (linhas={len(df_out)})")

if __name__ == "__main__":
    main()
