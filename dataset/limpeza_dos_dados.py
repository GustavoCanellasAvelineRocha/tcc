import os
from pathlib import Path
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

BASE_DIR = Path(__file__).parent.resolve()
INPUT_FILES = [
     BASE_DIR / "VALE3_dados_b3.csv",
     BASE_DIR / "ABEV3_dados_b3.csv",
     BASE_DIR / "ITUB4_dados_b3.csv",
]

PERIOD = 30
MODEL  = "multiplicative"

def build_components_from_b3(path_in: Path,
                             period: int = PERIOD,
                             model: str = MODEL) -> pd.DataFrame:

    # ======== LEITURA E LIMPEZA INICIAL ========
    df_raw = pd.read_csv(path_in, parse_dates=["data"], index_col="data")

    # (1) Remover duplicatas de data
    df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

    # (2) Garantir frequência uniforme (somente dias úteis)
    df_raw = df_raw.asfreq("B")

    # (3) Remover valores inválidos (0 ou negativos)
    df_raw = df_raw[df_raw["fechamento"] > 0]

    # (4) Converter para float e ordenar por data
    ser = df_raw["fechamento"].astype(float).sort_index()
    ser.name = "preco"

    # ======== DECOMPOSIÇÃO ========
    dec = seasonal_decompose(
        ser,
        period=period,
        model=model,
        two_sided=True,
        extrapolate_trend=False
    )

    tendencia    = dec.trend.rename("tendencia")
    sazonalidade = dec.seasonal.rename("sazonalidade")
    residuos     = dec.resid.rename("residuos")

    # ======== DATASET FINAL ========
    df = pd.concat([ser, residuos, tendencia, sazonalidade], axis=1)

    # Criação das diferenças (lags differenciadas)
    for k in range(1, 6):
        df[f"diff_{k}"] = df["preco"].diff(k)

    # Remover NaN gerados pela decomposição + diferenças
    df = df.dropna().copy()
    df.index.name = "data"

    return df


def process_file(path_in: Path):
    ticker = path_in.stem.split("_")[0].upper()

    # ======== SALVAR COM NOME DIFERENTE ========
    out_csv = path_in.parent / f"df_{ticker.lower()}_cleansing.csv"

    df = build_components_from_b3(path_in)
    df.to_csv(out_csv, encoding="utf8")

    print(
        f"[OK] {ticker}: {out_csv.name} | "
        f"shape={df.shape} | {df.index[0].date()} -> {df.index[-1].date()}"
    )


# MAIN
if __name__ == "__main__":
    for f in INPUT_FILES:
        process_file(f)
