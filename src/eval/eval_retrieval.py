"""Compara as configurações de busca nas perguntas com evidência do golden set.

Configurações: só BM25, só denso (Titan), só denso (local), híbrida (Titan + BM25
com RRF), híbrida + re-ranking. Saída: tabela no terminal e JSON em eval/results/.

Uso: python -m src.eval.eval_retrieval
"""
import json
import time
from datetime import date

import numpy as np
import pandas as pd

from src.config import ROOT
from src.eval.check_golden import carregar
from src.eval.julgamentos import ler
from src.eval.metrics import (cobertura_julgada_at_k, mrr, precisao_at_k, recall_at_k,
                              recall_normalizado_at_k)
from src.retrieval.hybrid import Buscador
from src.retrieval.rerank import reordenar

K_CANDIDATOS = 30


def main() -> None:
    perguntas = [p for p in carregar(ROOT / "eval" / "golden_set.jsonl") if p["tipo"] in {"especifica", "filtro"}]
    j = ler()
    relevantes = {q: set(g[g.relevante == "1"].review_id) for q, g in j.groupby("pergunta_id")}
    julgados = {q: set(g.review_id) for q, g in j.groupby("pergunta_id")}

    titan, local = Buscador("titan"), Buscador("local")
    texto = titan.por_id.texto_original
    configs = {
        "BM25": lambda p, f: titan.lexical(p, K_CANDIDATOS, f),
        "Denso (Titan)": lambda p, f: titan.denso(p, K_CANDIDATOS, f),
        "Denso (local)": lambda p, f: local.denso(p, K_CANDIDATOS, f),
        "Híbrida (Titan + BM25)": lambda p, f: [r.review_id for r in titan.hibrida(p, K_CANDIDATOS, f, dedupe=False)],
        "Híbrida + rerank": lambda p, f: [rid for rid, _ in reordenar(
            p, [(r.review_id, texto[r.review_id]) for r in titan.hibrida(p, K_CANDIDATOS, f, dedupe=False)])],
    }

    linhas, por_pergunta = [], []
    for nome, buscar in configs.items():
        inicio = time.time()
        m = []
        for p in perguntas:
            rel, jul = relevantes.get(p["id"], set()), julgados.get(p["id"], set())
            ranking = buscar(p["pergunta"], p.get("filtros_esperados"))
            valores = {
                "Recall@10": recall_at_k(ranking, rel, 10), "Recall@20": recall_at_k(ranking, rel, 20),
                "RecallN@8": recall_normalizado_at_k(ranking, rel, 8), "RecallN@20": recall_normalizado_at_k(ranking, rel, 20),
                "MRR": mrr(ranking, rel), "Precisão@8": precisao_at_k(ranking, rel, 8, jul),
                "Julgado@8": cobertura_julgada_at_k(ranking, jul, 8),
            }
            m.append(valores)
            por_pergunta.append({"config": nome, "pergunta_id": p["id"], **valores})
        medias = pd.DataFrame(m).mean().to_dict()
        linhas.append({"Configuração": nome, **medias, "seg/pergunta": (time.time() - inicio) / len(perguntas)})
        print(f"{nome}: ok")

    tabela = pd.DataFrame(linhas).set_index("Configuração").round(3)
    print(f"\n{len(perguntas)} perguntas com evidência (recall relativo ao pool de julgamentos)\n")
    print(tabela.to_string())
    saida = ROOT / "eval" / "results"
    saida.mkdir(parents=True, exist_ok=True)
    arquivo = saida / f"{date.today():%Y-%m-%d}_recuperacao.json"
    arquivo.write_text(json.dumps({"n_perguntas": len(perguntas), "k_candidatos": K_CANDIDATOS,
                                   "resumo": tabela.reset_index().to_dict("records"),
                                   "por_pergunta": por_pergunta}, ensure_ascii=False, indent=1, default=float))
    print(f"\nsalvo em {arquivo.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
