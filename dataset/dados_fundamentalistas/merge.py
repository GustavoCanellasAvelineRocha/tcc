#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge diário automático: cleansing + fundamentalistas (VALE3, ABEV3, ITUB4)
Robusto a variações de cabeçalho e SEM parênteses nos nomes finais.
"""

import sys
import re
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).parent.resolve()
CLEANSING_DIR = BASE_DIR.parent  # lê direto de /dataset/
FUND_DIR = BASE_DIR / "dados_fundamentalistas_filtrados"
OUT_DIR = BASE_DIR / "dados_cleansing_fundidos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nomes finais (sem parênteses) que queremos no CSV fundido
FUND_CANON = ["P/VP", "LL", "P/L", "EBIT TTM", "ROE", "EBIT/P"]

TICKERS = ["VALE3", "ABEV3", "ITUB4"]

# =========================
# Helpers
# =========================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove qualquer conteúdo entre parênteses e espaços extras."""
    df = df.copy()
    df.columns = [re.sub(r"\s*\(.*?\)", "", str(col)).strip() for col in df.columns]
    return df

def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Garante uma coluna datetime 'time' (aceita 'time', 'data' ou 1ª coluna)."""
    for cand in ("time", "data", "Date", "date"):
        if cand in df.columns:
            s = pd.to_datetime(df[cand], errors="coerce")
            if s.notna().any():
                out = df.copy()
                out["time"] = s
                return out[out["time"].notna()]
    first = df.columns[0]
    s = pd.to_datetime(df[first], errors="coerce", dayfirst=True)
    out = df.copy()
    out["time"] = s
    return out[out["time"].notna()]

def canonicalize_fund_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa nomes, aceita sinônimos e renomeia para canônicos:
    P/VP, LL, P/L, EBIT TTM, ROE, EBIT/P
    """
    df = clean_column_names(df)

    # mapa de sinônimos em lowercase (sem parênteses)
    synonyms = {
        "p/vp": ["p/vp"],
        "ll": ["ll", "lucro líquido ttm", "lucro líquido", "lucro/prejuízo do período ttm"],
        "p/l": ["p/l"],
        "ebit ttm": ["ebit ttm", "ebit"],
        "roe": ["roe"],
        "ebit/p": ["ebit/p", "ebit yield"],
    }

    # constrói um rename de qualquer sinônimo encontrado -> canônico
    lower_cols = {str(c).lower(): str(c) for c in df.columns}
    rename_map = {}

    def find_and_map(target_key: str, canonical_name: str):
        for alias in synonyms[target_key]:
            if alias in lower_cols:
                rename_map[lower_cols[alias]] = canonical_name
                return True
        return False

    ok = True
    ok &= find_and_map("p/vp", "P/VP")
    ok &= find_and_map("ll", "LL")
    ok &= find_and_map("p/l", "P/L")
    ok &= find_and_map("ebit ttm", "EBIT TTM")
    ok &= find_and_map("roe", "ROE")
    ok &= find_and_map("ebit/p", "EBIT/P")

    if not ok:
        missing = [c for c in FUND_CANON if c not in set(rename_map.values())]
        raise KeyError(
            "Colunas de fundamentos não encontradas (após limpeza de parênteses). "
            f"Faltando: {missing}. Disponíveis: {list(df.columns)}"
        )

    df = df.rename(columns=rename_map)

    # mantém apenas time + canônicas (se existirem)
    keep = ["time"] + [c for c in FUND_CANON if c in df.columns]
    return df[keep]

# =========================
# Merge por ticker
# =========================
def merge_one_ticker(ticker: str):
    print(f"\n==============================")
    print(f"[INFO] Processando {ticker} ...")

    clean_path = CLEANSING_DIR / f"df_{ticker.lower()}_cleansing.csv"
    fund_path  = FUND_DIR / f"{ticker}_dados_fundamentalistas_diarios_final.csv"
    out_path   = OUT_DIR / f"{ticker}_cleansing_fundamentalista.csv"

    if not clean_path.exists():
        print(f"[ERRO] Arquivo cleansing não encontrado: {clean_path}")
        return
    if not fund_path.exists():
        print(f"[ERRO] Arquivo fundamentalista não encontrado: {fund_path}")
        return

    # --- Ler cleansing (PT-BR) ---
    df_clean = pd.read_csv(clean_path)
    if "data" not in df_clean.columns:
        raise ValueError(f"O arquivo cleansing {clean_path.name} deve ter coluna 'data'.")

    df_clean["data"] = pd.to_datetime(df_clean["data"], errors="coerce")
    df_clean = df_clean[df_clean["data"].notna()].rename(columns={"data": "time"})

    # opcional: padronizar nome do preço para 'preco'
    if "fechamento" in df_clean.columns and "preco" not in df_clean.columns:
        df_clean = df_clean.rename(columns={"fechamento": "preco"})

    # --- Ler fundamentos (aceita variações e remove parênteses) ---
    df_fund = pd.read_csv(fund_path)
    df_fund = ensure_time_column(df_fund)
    df_fund = canonicalize_fund_columns(df_fund)  # agora tem time + canônicas SEM parênteses

    # Merge left
    merged = pd.merge(df_clean, df_fund, on="time", how="left") \
               .sort_values("time").reset_index(drop=True)

    # Forward fill nos fundamentos
    fund_cols = [c for c in FUND_CANON if c in merged.columns]
    merged[fund_cols] = merged[fund_cols].ffill()

    # Salvar CSV final
    merged.to_csv(out_path, index=False)
    print(f"[OK] Merge concluído e salvo em: {out_path}")

    # Stats rápidas
    print("\n--- Head ---")
    print(merged.head(3).to_string(index=False))
    print("\n--- Tail ---")
    print(merged.tail(3).to_string(index=False))
    if fund_cols:
        print("\n--- Sanidade (indicadores) ---")
        with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
            print(merged[fund_cols].describe())
    else:
        print("\n[AVISO] Nenhuma coluna de fundamentos encontrada após o merge.")

# =========================
# MAIN
# =========================
def main():
    for t in TICKERS:
        try:
            merge_one_ticker(t)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {t}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
