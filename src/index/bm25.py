"""Índice lexical BM25 sobre o texto normalizado.

A tokenização é própria do BM25 (não vale para os embeddings): minúsculas,
sem acento, sem stop words, mas MANTENDO as negações. Em avaliações, "não
chegou" e "chegou" são opostos; se o "não" sumisse, a busca juntaria os dois.
"""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from unidecode import unidecode

NEGACOES = {"nao", "nunca", "nem", "nenhum", "nenhuma", "jamais", "sem"}

# Stop words do português (lista do NLTK, já sem acento), menos as negações acima.
_STOP = """a ao aos aquela aquelas aquele aqueles aquilo as ate com como da das de dela delas dele deles
depois do dos e ela elas ele eles em entre era eram essa essas esse esses esta estas este estes eu foi
foram ha isso isto ja lhe lhes mais mas me mesmo meu meus minha minhas muito na nas no nos nossa nossas
nosso nossos num numa o os ou para pela pelas pelo pelos por qual quando que quem se seja ser seu seus so
sua suas tambem te tem tinha tu tua tuas um uma voce voces vos estou esta estamos estao estive esteve
houve ha havia seria serao sao sou somos fui fomos""".split()
STOP_WORDS = set(_STOP) - NEGACOES

_TOKEN = re.compile(r"\w+")


def tokenizar(texto: str) -> list[str]:
    tokens = _TOKEN.findall(unidecode(texto.lower()))
    return [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]


class IndiceBM25:
    def __init__(self, ids: list[str], textos: list[str]):
        self.ids = list(ids)
        self.bm25 = BM25Okapi([tokenizar(t) for t in textos])

    def buscar(self, pergunta: str, k: int = 20, mascara=None) -> list[tuple[str, float]]:
        """Top-k (review_id, score). `mascara` (array de bool) restringe a busca a um recorte."""
        scores = self.bm25.get_scores(tokenizar(pergunta))
        if mascara is not None:
            scores = scores.copy()
            scores[~mascara] = -1.0
        ordem = scores.argsort()[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in ordem if scores[i] > 0]

    def salvar(self, arquivo: Path) -> None:
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        with open(arquivo, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def carregar(arquivo: Path) -> "IndiceBM25":
        with open(arquivo, "rb") as f:
            return pickle.load(f)
