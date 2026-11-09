"""Filtros de metadados: o mesmo recorte vira `where` do Chroma e máscara do BM25.

Formato do filtro (dict): igualdade simples ({"categoria_pt": "telefonia"}) ou
limites com sufixo _min/_max ({"review_score_max": 2, "valor_min": 500}).
"valor_min"/"valor_max" se referem a valor_total.
"""
import numpy as np
import pandas as pd

APELIDOS = {"valor": "valor_total"}


def _campo(chave: str) -> tuple[str, str]:
    for sufixo, op in (("_min", "$gte"), ("_max", "$lte")):
        if chave.endswith(sufixo):
            base = chave[: -len(sufixo)]
            return APELIDOS.get(base, base), op
    return chave, "$eq"


def _valor(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def para_chroma(filtros: dict | None) -> dict | None:
    if not filtros:
        return None
    condicoes = [{_campo(k)[0]: {_campo(k)[1]: _valor(v)}} for k, v in filtros.items()]
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


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
