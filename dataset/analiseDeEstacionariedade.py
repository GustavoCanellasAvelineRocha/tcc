import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

datasets = {
    "VALE3": os.path.join(BASE_DIR, "df_vale3_cleansing.csv"),
    "ABEV3": os.path.join(BASE_DIR, "df_abev3_cleansing.csv"),
    "ITUB4": os.path.join(BASE_DIR, "df_itub4_cleansing.csv"),
}

def main():
    print("Iniciando Análise de estacionariedade")
    saida = [f"Análise"]

    for ativo, caminho in datasets.items():
        df = pd.read_csv(caminho, index_col=0, parse_dates=True)
        print(f"\n--- Analisando o {ativo} ---")
        saida.append(f"\n--- Analisando o {ativo} ---\n")

        serie = df["preco"].dropna()
        resultado = adfuller(serie, autolag="AIC")
        estatistica, pvalor, n_lags, n_obs = resultado[0:4]

        if pvalor < 0.05:
            decisao = "Estacionária pois rejeita H0"
        else:
            decisao = "Não estacionária, pois não rejeita H0"

        print("")
        print(f"--- {ativo}, com variável alvo = preco")
        print(f"Estatística ADF: {estatistica:.4f}")
        print(f"p-valor: {pvalor:.4f}")
        print(f"Nº lags: {n_lags}, Nº observações: {n_obs}")
        print(f"Decisão: {decisao}")

        bloco = (
            f"--- {ativo}, com variável alvo = preco\n"
            f"Estatística ADF: {estatistica:.4f}\n"
            f"p-valor: {pvalor:.4f}\n"
            f"Nº lags: {n_lags}, Nº observações: {n_obs}\n"
            f"Decisão: {decisao}\n"
        )
        saida.append(bloco)

    # salva o relatório consolidado
    relatorio_path = os.path.join(BASE_DIR, "analise_de_estacionariedade.txt")
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.writelines(saida)

    print(f"\nAnálise salva com o nome de '{relatorio_path}'")

if __name__ == "__main__":
    main()
