import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

COL_PRECO = "preco"   

files = {
    "ABEV3": BASE_DIR / "df_abev3_cleansing.csv",
    "ITUB4": BASE_DIR / "df_itub4_cleansing.csv",
    "VALE3": BASE_DIR / "df_vale3_cleansing.csv",
}

saida = []

for nome, caminho in files.items():
    df = pd.read_csv(caminho)

    serie = df[COL_PRECO]

    saida.append(f"===== {nome} =====\n")
    saida.append(f"Série: {COL_PRECO}")
    saida.append(f"  Média:   {serie.mean():.4f}")
    saida.append(f"  Mediana: {serie.median():.4f}")
    saida.append(f"  Moda:    {serie.mode().iloc[0]:.4f}")
    saida.append(f"  Mínimo:  {serie.min():.4f}")
    saida.append(f"  Máximo:  {serie.max():.4f}")
    saida.append("")  

output_file = BASE_DIR / "estatisticas_preco.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(saida))

print(f"Arquivo salvo em: {output_file}")
