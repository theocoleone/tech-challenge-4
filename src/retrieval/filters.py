"""Filtros de metadados: viram uma máscara booleana sobre documents.parquet,
aplicada antes da busca vetorial e da busca BM25.

Formato do filtro (dict): igualdade simples ({"categoria_pt": "telefonia"}) ou
limites com sufixo _min/_max ({"review_score_max": 2, "valor_min": 500}).
"valor_min"/"valor_max" se referem a valor_total.
"""
import numpy as np
import pandas as pd

APELIDOS = {"valor": "valor_total"}


def _campo(chave: str) -> tuple[str, str]:
    """"review_score_max" -> ("review_score", "$lte"); "categoria_pt" -> ("categoria_pt", "$eq")."""
    for sufixo, op in (("_min", "$gte"), ("_max", "$lte")):
        if chave.endswith(sufixo):
            base = chave[: -len(sufixo)]
            return APELIDOS.get(base, base), op
    return chave, "$eq"


def mascara(docs: pd.DataFrame, filtros: dict | None) -> np.ndarray:
    m = np.ones(len(docs), dtype=bool)
    for k, v in (filtros or {}).items():
        campo, op = _campo(k)
        col = docs[campo]
        if op == "$gte":
            m &= (col >= v).fillna(False).to_numpy()
        elif op == "$lte":
            m &= (col <= v).fillna(False).to_numpy()
        else:
            m &= (col == v).fillna(False).to_numpy()
    return m
