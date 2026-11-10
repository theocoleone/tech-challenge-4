"""Constrói os índices de busca a partir de data/processed/documents.parquet.

- index/bm25.pkl: índice lexical (BM25).
- index/vetores_<modelo>.npy: vetores normalizados de cada avaliação, na mesma
  ordem de documents.parquet (os metadados para filtrar vêm do próprio parquet).

Os embeddings ficam em cache (index/embeddings_<modelo>.parquet): refazer o índice
não chama o Bedrock de novo.

Uso:
  python -m src.index.build_index                 # titan + local + bm25
  python -m src.index.build_index --modelos titan
"""
import argparse

import pandas as pd

from src.config import get_settings
from src.index.bm25 import IndiceBM25
from src.index.embeddings import embeddings_com_cache
from src.index.vector_store import IndiceVetorial


def build(modelos: list[str]) -> None:
    s = get_settings()
    docs = pd.read_parquet(s.data_processed / "documents.parquet")
    ids, textos = docs.review_id.tolist(), docs.texto_normalizado.tolist()
    print(f"{len(docs):,} documentos")

    IndiceBM25(ids, textos).salvar(s.index_dir / "bm25.pkl")
    print("BM25: ok")

    for modelo in modelos:
        vetores = embeddings_com_cache(modelo, ids, textos, s.index_dir)
        IndiceVetorial(ids, vetores).salvar(s.index_dir, modelo)
        print(f"vetores {modelo}: {vetores.shape[0]:,} x {vetores.shape[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["titan", "local"])
    build(ap.parse_args().modelos)
