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
COLUNAS = ["pergunta_id", "pergunta", "review_id", "nota", "categoria", "pedido", "texto", "relevante", "encontrado_por"]


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


def descrever_pedido(d) -> str:
    """Recorte de cada avaliação, para julgar perguntas com filtro (mês, valor, status, atraso)."""
    atraso = "com atraso" if d.entregue_com_atraso else ("no prazo" if d.entregue else "não entregue")
    return f"compra {str(d.ano_mes_compra)[:4]}-{str(d.ano_mes_compra)[4:]} | R$ {d.valor_total:.0f} | {d.order_status} | {atraso}"


def escrever(df: pd.DataFrame) -> None:
    if "pedido" not in df or (df["pedido"].fillna("") == "").any():
        from src.config import get_settings

        docs = pd.read_parquet(get_settings().data_processed / "documents.parquet").set_index("review_id")
        df = df.copy()
        df["pedido"] = [descrever_pedido(docs.loc[r]) if r in docs.index else "" for r in df.review_id]
    df = df[COLUNAS]
    df.to_csv(CSV, index=False)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="julgamentos")
        ws = w.sheets["julgamentos"]
        larguras = {"A": 11, "B": 45, "C": 14, "D": 6, "E": 22, "F": 34, "G": 90, "H": 11, "I": 18}
        for col, largura in larguras.items():
            ws.column_dimensions[col].width = largura
        for cel in ws[1]:
            cel.font = Font(bold=True)
        for linha in ws.iter_rows(min_row=2):
            for cel in linha:
                cel.alignment = Alignment(wrap_text=cel.column_letter in ("B", "F", "G"), vertical="top")
            linha[7].fill = PatternFill("solid", fgColor="FFF2CC")  # coluna a preencher
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.column_dimensions["I"].hidden = True  # quem encontrou fica oculto para não influenciar
        dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
        dv.add(f"H2:H{len(df) + 1}")
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
