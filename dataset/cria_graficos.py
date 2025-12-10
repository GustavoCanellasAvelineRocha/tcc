import os
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

arquivos = {
    "VALE3": "df_vale3_cleansing.csv",
    "ITUB4": "df_itub4_cleansing.csv",
    "ABEV3": "df_abev3_cleansing.csv",
}

OUTPUT_DIR = os.path.join(CURRENT_DIR, "graficos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for ticker, filename in arquivos.items():
    csv_path = os.path.join(CURRENT_DIR, filename)

    df = (
        pd.read_csv(csv_path, parse_dates=["data"])
        .sort_values("data")
        .set_index("data")
    )

    n_total = len(df)
    n_train = int(n_total * 0.8)

    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    # ----------- Gráfico Único ----------- #
    plt.figure(figsize=(14, 5))

    # Série completa
    plt.plot(df.index, df["preco"], color="gray", alpha=0.4, label="Série inteira")

    # Parte de treino (80%)
    plt.plot(df_train.index, df_train["preco"], color="blue", label="80% iniciais (treino)")

    # Parte de teste (20%)
    plt.plot(df_test.index, df_test["preco"], color="orange", label="20% finais (teste)")

    plt.title(f"{ticker} - Série com split 80/20")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_serie_80_20.png"), dpi=300)
    plt.close()

    print(f"Gráfico único da ação {ticker} salvo com sucesso.")
