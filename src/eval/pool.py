"""Monta a planilha de julgamentos de relevância (pooling).

Para cada pergunta com evidência e com filtro do golden set, junta o top-20 de
três buscadores diferentes (BM25, Titan e modelo local). O Theo marca cada par
pergunta x avaliação como relevante (1) ou não (0), na coluna "relevante". Com isso dá para medir
recall e MRR de qualquer configuração depois, sem rotular a base inteira.

O recall medido é relativo a esse conjunto ("pool"): uma avaliação relevante que
nenhum dos três encontrou não entra na conta. Isso é declarado no README.

Uso: python -m src.eval.pool   (gera eval/julgamentos.xlsx e .csv; não apaga marcações)
"""
import random

import chromadb
import pandas as pd

from src.config import ROOT, get_settings
from src.data.clean import normalizar
from src.eval.check_golden import carregar
from src.eval.julgamentos import escrever, ler
from src.index.bm25 import IndiceBM25
from src.index.build_index import colecao
from src.index.embeddings import get_embedder
from src.retrieval.filters import mascara, para_chroma

K = 20
TIPOS = {"especifica", "filtro"}


def main() -> None:
    s = get_settings()
    docs = pd.read_parquet(s.data_processed / "documents.parquet").set_index("review_id", drop=False)
    perguntas = [p for p in carregar(ROOT / "eval" / "golden_set.jsonl") if p["tipo"] in TIPOS]
    bm25 = IndiceBM25.carregar(s.index_dir / "bm25.pkl")
    cliente = chromadb.PersistentClient(path=str(s.index_dir / "chroma"))
    densos = {m: (get_embedder(m), cliente.get_collection(colecao(m))) for m in ("titan", "local")}

    linhas = []
    for p in perguntas:
        consulta = normalizar(p["pergunta"])
        filtros = p.get("filtros_esperados")
        fontes: dict[str, set[str]] = {}
        for rid, _ in bm25.buscar(consulta, K, mascara(docs, filtros)):
            fontes.setdefault(rid, set()).add("bm25")
        for nome, (emb, col) in densos.items():
            r = col.query(query_embeddings=[emb.embed_query(consulta)], n_results=K, where=para_chroma(filtros))
            for rid in r["ids"][0]:
                fontes.setdefault(rid, set()).add(nome)
        ids = list(fontes)
        random.Random(p["id"]).shuffle(ids)  # embaralha para a ordem não sugerir qual buscador achou
        for rid in ids:
            d = docs.loc[rid]
            linhas.append({
                "pergunta_id": p["id"], "pergunta": p["pergunta"], "review_id": rid,
                "nota": int(d.review_score), "categoria": d.categoria_pt, "texto": d.texto_original,
                "relevante": "", "encontrado_por": ",".join(sorted(fontes[rid])),
            })
        print(f"{p['id']}: {len(ids)} avaliações para julgar")

    novo = pd.DataFrame(linhas)
    antigo = ler()
    if len(antigo):  # preserva o que já foi marcado
        marcados = antigo[antigo.relevante != ""].set_index(["pergunta_id", "review_id"]).relevante
        novo["relevante"] = [marcados.get(c, "") for c in zip(novo.pergunta_id, novo.review_id)]
    escrever(novo)
    print(f"\n{len(novo)} pares em eval/julgamentos.xlsx ({(novo.relevante != '').sum()} já marcados)")

if __name__ == "__main__":
    main()
