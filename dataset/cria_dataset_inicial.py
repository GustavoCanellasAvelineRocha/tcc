# -*- coding: utf-8 -*-
from datetime import datetime
from io import BytesIO
from pathlib import Path
import zipfile
import requests
import pandas as pd

BASE_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
TICKERS = {"ITUB4", "ABEV3", "VALE3"}

START_DATE = pd.Timestamp("2015-06-10")
END_DATE = pd.Timestamp("2025-06-10")
OUTPUT_DIR = Path(__file__).parent.resolve()

def parse_cotahist(raw_txt: bytes,
                   tickers_set: set,
                   start_date: pd.Timestamp,
                   end_date: pd.Timestamp) -> pd.DataFrame:
    records = []
    for line in raw_txt.splitlines():
        if not line or len(line) < 245 or line[0:2] != b"01":
            continue

        date_str = line[2:10].decode("ascii")        
        ticker = line[12:24].decode("ascii").strip()
        market_type = line[24:27].decode("ascii")
        if ticker not in tickers_set or market_type != "010":
            continue

        date = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
        if date < start_date or date > end_date:
            continue

        def _price(s): return int(s.decode("ascii")) / 100.0
        def _int(s): return int(s.decode("ascii"))

        open_price = _price(line[56:69])
        high_price = _price(line[69:82])
        low_price = _price(line[82:95])
        close_price = _price(line[108:121])
        volume = _int(line[152:170])
        factor = int(line[210:217].decode("ascii")) or 1

        open_price /= factor
        high_price /= factor
        low_price /= factor
        close_price /= factor

        records.append((date, ticker, open_price, high_price, low_price, close_price, volume))

    if not records:
        return pd.DataFrame(columns=["data", "ticker", "abertura", "maximo", "minimo", "fechamento", "volume"])

    df = pd.DataFrame(
        records,
        columns=["data", "ticker", "abertura", "maximo", "minimo", "fechamento", "volume"]
    )
    return df.sort_values(["ticker", "data"]).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted({d.year for d in pd.date_range(START_DATE, END_DATE, freq="D")})
    all_dfs = []

    for year in years:
        url = BASE_URL.format(year=year)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        blob = r.content

        with zipfile.ZipFile(BytesIO(blob)) as zf:
            txt_name = next((n for n in zf.namelist() if n.upper().endswith(".TXT")), None)
            if txt_name is None:
                print(f"[WARNING] ZIP {year} without expected TXT file.")
                continue
            part = parse_cotahist(zf.read(txt_name), TICKERS, START_DATE, END_DATE)
            if not part.empty:
                all_dfs.append(part)

    data = pd.concat(all_dfs, ignore_index=True)
    for ticker in sorted(data["ticker"].unique()):
        df_ticker = (
            data.loc[data["ticker"] == ticker, ["data", "abertura", "maximo", "minimo", "fechamento", "volume"]]
                .sort_values("data")
                .reset_index(drop=True)
        )
        out_path = OUTPUT_DIR / f"{ticker}_dados_b3.csv"
        df_ticker.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"{ticker}: Operação de criação do dataset finalizada e salva como {ticker}_dados_b3.csv")

if __name__ == "__main__":
    main()