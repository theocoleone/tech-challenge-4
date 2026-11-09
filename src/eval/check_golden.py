"""Confere os rótulos do golden set e do conjunto de calibração contra a base.

Para as perguntas marcadas como "sem evidência" ou "evidência insuficiente",
procura os termos de verificação no texto das avaliações:
- sem_evidencia precisa ter 0 ocorrências;
- evidencia_fraca precisa ter poucas (1 a 5) ou um recorte pequeno.
Para "filtro sem resultado", confere que o filtro não devolve nenhuma avaliação.

Uso: python -m src.eval.check_golden
"""
import json
import re
from pathlib import Path

import pandas as pd

from src.config import ROOT, get_settings


def carregar(arquivo: Path) -> list[dict]:
    return [json.loads(l) for l in arquivo.read_text().splitlines() if l.strip()]


def aplicar_filtros(docs: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    m = pd.Series(True, index=docs.index)
    for chave, valor in filtros.items():
        if chave.endswith("_min"):
            col = "valor_total" if chave == "valor_min" else chave[:-4]
            m &= docs[col] >= valor
        elif chave.endswith("_max"):
            col = "valor_total" if chave == "valor_max" else chave[:-4]
            m &= docs[col] <= valor
        else:
            m &= docs[chave] == valor
    return docs[m]


def conferir(perguntas: list[dict], docs: pd.DataFrame) -> list[str]:
    texto = docs.texto_normalizado.str.lower()
    problemas = []
    for p in perguntas:
        status = p["status_esperado"]
        if p.get("termos_verificacao"):
            n = int(texto.str.contains("|".join(p["termos_verificacao"]), regex=True).sum())
            p["_ocorrencias"] = n
            if status == "sem_evidencia" and n > 0:
                problemas.append(f"{p['id']}: rotulada sem evidência, mas {n} avaliações citam os termos")
            if status == "evidencia_fraca" and not 1 <= n <= 5:
                problemas.append(f"{p['id']}: rotulada evidência fraca, mas {n} avaliações citam os termos")
        if p.get("filtros_esperados"):
            n = len(aplicar_filtros(docs, p["filtros_esperados"]))
            p["_no_recorte"] = n
            if status == "filtro_sem_resultado" and n > 0:
                problemas.append(f"{p['id']}: rotulada filtro sem resultado, mas o filtro devolve {n}")
            if status == "ok" and n == 0:
                problemas.append(f"{p['id']}: filtro vazio numa pergunta que deveria ter resposta")
    return problemas


def main() -> None:
    docs = pd.read_parquet(get_settings().data_processed / "documents.parquet")
    for nome in ("golden_set", "calibracao"):
        perguntas = carregar(ROOT / "eval" / f"{nome}.jsonl")
        problemas = conferir(perguntas, docs)
        print(f"\n{nome}: {len(perguntas)} perguntas")
        for p in perguntas:
            extra = [f"{k[1:]}={p[k]}" for k in ("_ocorrencias", "_no_recorte") if k in p]
            if extra:
                print(f"  {p['id']} [{p['status_esperado']}] {', '.join(extra)}")
        print("  OK" if not problemas else "\n".join("  PROBLEMA " + x for x in problemas))


if __name__ == "__main__":
    main()
