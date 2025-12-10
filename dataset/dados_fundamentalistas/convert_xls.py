# convert_xls_via_excel_auto.py
# Converte todos os arquivos .xls da pasta "balancos/" em .xlsx,
# usando o Excel via COM (Windows), sem argumentos de linha de comando.

import os
from pathlib import Path
import re
import sys
import time

try:
    import win32com.client as win32
except Exception:
    print("[ERRO] pywin32 não está instalado. Rode: pip install pywin32")
    sys.exit(1)


def sanitize(name: str) -> str:
    """Remove caracteres inválidos do nome da planilha."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name or "Sheet"


def convert_xls_via_excel(xls_path: Path, outdir: Path):
    """Abre um .xls no Excel e exporta todas as abas como .xlsx"""
    excel = None
    try:
        excel = win32.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        wb = excel.Workbooks.Open(str(xls_path.resolve()))
        base = xls_path.stem

        for ws in wb.Worksheets:
            ws.Copy()
            newwb = excel.ActiveWorkbook
            sheet_name = sanitize(ws.Name)
            base_out = f"{base}__{sheet_name}"

            xlsx_path = outdir / f"{base_out}.xlsx"
            newwb.SaveAs(str(xlsx_path.resolve()), FileFormat=51)  # .xlsx
            newwb.Close(SaveChanges=False)

        wb.Close(SaveChanges=False)

    finally:
        if excel is not None:
            excel.DisplayAlerts = False
            excel.Quit()
            time.sleep(0.5)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    in_dir = base_dir / "balancos"
    out_dir = base_dir / "convertido"
    out_dir.mkdir(parents=True, exist_ok=True)

    xls_files = list(in_dir.glob("*.xls"))
    if not xls_files:
        print(f"[ERRO] Nenhum .xls encontrado em {in_dir}")
        sys.exit(1)

    print(f"[INFO] Encontrados {len(xls_files)} arquivos .xls para converter.")
    for xls_path in xls_files:
        print(f"[INFO] Convertendo: {xls_path.name} -> {out_dir}")
        convert_xls_via_excel(xls_path, out_dir)
        print(f"[OK] {xls_path.name} convertido com sucesso.\n")

    print("[✔] Conversão concluída para todos os arquivos .xlsx.")
