"""Recuperação: busca vetorial, busca lexical (BM25) e a fusão das duas (RRF).

A busca vetorial encontra avaliações pelo significado ("demorou demais" para uma
pergunta sobre atraso). O BM25 encontra pela palavra exata, o que ajuda com termos
específicos ("nota fiscal", "voltagem"). A fusão por Reciprocal Rank Fusion
combina as duas listas usando só a posição de cada avaliação em cada lista, sem
precisar comparar escalas de score diferentes.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.clean import normalizar
from src.index.bm25 import IndiceBM25
from src.index.embeddings import get_embedder
from src.index.vector_store import IndiceVetorial
from src.retrieval.filters import mascara

RRF_K = 60  # constante padrão do RRF (Cormack et al., 2009)


@dataclass
class Resultado:
    review_id: str
    score: float
    origem: dict = field(default_factory=dict)  # posição em cada lista (para explicar o ranking)


def rrf(listas: dict[str, list[str]], k: int = RRF_K) -> list[Resultado]:
    """Soma 1/(k + posição) de cada lista. Quem aparece bem nas duas sobe."""
    acumulado: dict[str, Resultado] = {}
    for nome, ids in listas.items():
        for pos, rid in enumerate(ids, start=1):
            r = acumulado.setdefault(rid, Resultado(rid, 0.0))
            r.score += 1.0 / (k + pos)
            r.origem[nome] = pos
    return sorted(acumulado.values(), key=lambda r: r.score, reverse=True)


class Buscador:
    def __init__(self, modelo_denso: str = "titan"):
        s = get_settings()
        self.docs = pd.read_parquet(s.data_processed / "documents.parquet")
        self.por_id = self.docs.set_index("review_id")
        self.bm25 = IndiceBM25.carregar(s.index_dir / "bm25.pkl")
        self.modelo_denso = modelo_denso
        self.embedder = get_embedder(modelo_denso)
        self.vetorial = IndiceVetorial.carregar(s.index_dir, modelo_denso)

    def lexical(self, consulta: str, k: int = 30, filtros: dict | None = None) -> list[str]:
        return [rid for rid, _ in self.bm25.buscar(normalizar(consulta), k, mascara(self.docs, filtros))]

    def denso(self, consulta: str, k: int = 30, filtros: dict | None = None) -> list[str]:
        vetor = self.embedder.embed_query(normalizar(consulta))
        return [rid for rid, _ in self.vetorial.buscar(vetor, k, mascara(self.docs, filtros) if filtros else None)]

    def hibrida(self, consulta: str, k: int = 30, filtros: dict | None = None, dedupe: bool = True) -> list[Resultado]:
        fundidos = rrf({"denso": self.denso(consulta, k, filtros), "bm25": self.lexical(consulta, k, filtros)})
        return self.deduplicar(fundidos) if dedupe else fundidos

    def vetores(self, ids: list[str]) -> np.ndarray:
        """Vetores já calculados das avaliações (não chama o Bedrock de novo)."""
        return self.vetorial.vetores_de(ids)

    def deduplicar(self, resultados: list[Resultado]) -> list[Resultado]:
        """Textos idênticos ("muito bom", "atraso na entrega") ocupariam várias posições:
        fica só a primeira ocorrência de cada texto."""
        vistos, saida = set(), []
        for r in resultados:
            chave = self.por_id.loc[r.review_id, "texto_normalizado"].lower()
            if chave not in vistos:
                vistos.add(chave)
                saida.append(r)
        return saida
