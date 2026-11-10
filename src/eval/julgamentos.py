"""Leitura e escrita da planilha de julgamentos de relevância.

O Theo marca em eval/julgamentos.xlsx (abre certo no Excel em qualquer idioma).
eval/julgamentos.csv é a cópia em texto simples, versionada no git e lida pelas
métricas. `make julgamentos` atualiza o CSV a partir do Excel.
"""
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from src.config import ROOT

XLSX = ROOT / "eval" / "julgamentos.xlsx"
CSV = ROOT / "eval" / "julgamentos.csv"
COLUNAS = ["pergunta_id", "pergunta", "review_id", "nota", "categoria", "texto", "relevante", "encontrado_por"]


def ler() -> pd.DataFrame:
    """Lê as marcações: prefere o Excel (onde o Theo marca), senão o CSV."""
    if XLSX.exists():
        df = pd.read_excel(XLSX, dtype=str)
    elif CSV.exists():
        df = pd.read_csv(CSV, dtype=str)
    else:
        return pd.DataFrame(columns=COLUNAS)
    df = df.fillna("")
    for col in COLUNAS:
        if col not in df:
            df[col] = ""
    df["relevante"] = df["relevante"].str.strip().str.replace(r"\.0$", "", regex=True)
    return df[COLUNAS]


def escrever(df: pd.DataFrame) -> None:
    df = df[COLUNAS]
    df.to_csv(CSV, index=False)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="julgamentos")
        ws = w.sheets["julgamentos"]
        larguras = {"A": 11, "B": 45, "C": 14, "D": 6, "E": 22, "F": 90, "G": 11, "H": 18}
        for col, largura in larguras.items():
            ws.column_dimensions[col].width = largura
        for cel in ws[1]:
            cel.font = Font(bold=True)
        for linha in ws.iter_rows(min_row=2):
            for cel in linha:
                cel.alignment = Alignment(wrap_text=cel.column_letter in ("B", "F"), vertical="top")
            linha[6].fill = PatternFill("solid", fgColor="FFF2CC")  # coluna a preencher
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.column_dimensions["H"].hidden = True  # quem encontrou fica oculto para não influenciar
        dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
        dv.add(f"G2:G{len(df) + 1}")
        ws.add_data_validation(dv)


def sincronizar_csv() -> None:
    df = ler()
    df.to_csv(CSV, index=False)
    marcados = (df.relevante != "").sum()
    invalidos = df[~df.relevante.isin(["", "0", "1"])]
    print(f"{CSV.relative_to(ROOT)} atualizado: {marcados}/{len(df)} marcados")
    if len(invalidos):
        print(f"ATENÇÃO: {len(invalidos)} valores diferentes de 0/1, por exemplo: {invalidos.relevante.unique()[:5]}")


if __name__ == "__main__":
    sincronizar_csv()
