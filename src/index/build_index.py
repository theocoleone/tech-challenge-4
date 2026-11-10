"""Constrói os índices de busca a partir de data/processed/documents.parquet.

- index/bm25.pkl: índice lexical (BM25).
- index/chroma/: banco vetorial Chroma, uma coleção por modelo de embedding,
  com distância de cosseno e os metadados de cada avaliação para os filtros.

Uso:
  python -m src.index.build_index                 # titan + local + bm25
  python -m src.index.build_index --modelos titan
"""
import argparse

import chromadb
import numpy as np
import pandas as pd

from src.config import get_settings
from src.index.bm25 import IndiceBM25
from src.index.embeddings import embeddings_com_cache

METADADOS = [
    "review_score", "data_avaliacao", "data_envio_pesquisa", "ano_mes_compra", "order_status",
    "valor_total", "multi_pedido", "entregue", "dias_atraso", "entregue_com_atraso",
    "categoria_pt", "categoria_en", "multi_categoria", "n_itens", "uf_cliente", "n_palavras", "order_ids",
]


def metadados_chroma(linha: dict) -> dict:
    """O Chroma só aceita str, int, float e bool, e não aceita valor vazio: chaves vazias são omitidas."""
    saida = {}
    for chave in METADADOS:
        valor = linha[chave]
        if valor is None or pd.isna(valor):
            continue
        if isinstance(valor, (bool, np.bool_)):
            saida[chave] = bool(valor)
        elif isinstance(valor, (int, np.integer)):
            saida[chave] = int(valor)
        elif isinstance(valor, (float, np.floating)):
            saida[chave] = float(valor)
        else:
            saida[chave] = str(valor)
    return saida


def colecao(modelo: str) -> str:
    return f"reviews_{modelo}"


def build(modelos: list[str]) -> None:
    s = get_settings()
    docs = pd.read_parquet(s.data_processed / "documents.parquet")
    ids, textos = docs.review_id.tolist(), docs.texto_normalizado.tolist()
    print(f"{len(docs):,} documentos")

    IndiceBM25(ids, textos).salvar(s.index_dir / "bm25.pkl")
    print("BM25: ok")

    cliente = chromadb.PersistentClient(path=str(s.index_dir / "chroma"))
    metas = [metadados_chroma(r) for r in docs.to_dict("records")]
    for modelo in modelos:
        vetores = embeddings_com_cache(modelo, ids, textos, s.index_dir)
        nome = colecao(modelo)
        if nome in [c.name for c in cliente.list_collections()]:
            cliente.delete_collection(nome)  # o índice é sempre reconstruído do zero a partir do cache
        col = cliente.create_collection(nome, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(ids), 5000):
            col.add(ids=ids[i:i + 5000], embeddings=vetores[i:i + 5000].tolist(),
                    documents=textos[i:i + 5000], metadatas=metas[i:i + 5000])
        print(f"Chroma {nome}: {col.count():,} vetores de dimensão {vetores.shape[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["titan", "local"])
    build(ap.parse_args().modelos)
