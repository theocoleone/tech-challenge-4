"""Índice vetorial com busca exata (força bruta) em NumPy.

Com 42 mil avaliações e vetores de 1.024 dimensões, comparar a pergunta com todos
os vetores é uma multiplicação de matriz de ~170 MB que leva poucos milissegundos.
Nessa escala não é preciso um índice aproximado (HNSW/IVF): a busca exata é mais
simples, não perde resultados e permite filtrar por metadados antes de comparar.

Começamos com o Chroma, mas descartamos: o banco dele é um SQLite, e o software de
backup corporativo da máquina de desenvolvimento abria esse arquivo e segurava a
trava, o que deixava a abertura do Chroma parada por minutos. Os vetores aqui são
um arquivo .npy comum, lido sem trava nenhuma.
"""
import json
from pathlib import Path

import numpy as np


class IndiceVetorial:
    def __init__(self, ids: list[str], vetores: np.ndarray):
        self.ids = list(ids)
        v = np.asarray(vetores, dtype=np.float32)
        self.vetores = v / np.linalg.norm(v, axis=1, keepdims=True)  # cosseno = produto interno
        self.posicao = {rid: i for i, rid in enumerate(self.ids)}

    def buscar(self, consulta: np.ndarray, k: int = 30, mascara: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Top-k (review_id, similaridade de cosseno), opcionalmente só dentro de um recorte."""
        q = np.asarray(consulta, dtype=np.float32)
        q = q / np.linalg.norm(q)
        candidatos = np.arange(len(self.ids)) if mascara is None else np.flatnonzero(mascara)
        if len(candidatos) == 0:
            return []
        sims = self.vetores[candidatos] @ q
        k = min(k, len(candidatos))
        topo = np.argpartition(-sims, k - 1)[:k]
        topo = topo[np.argsort(-sims[topo])]
        return [(self.ids[candidatos[i]], float(sims[i])) for i in topo]

    def vetores_de(self, ids: list[str]) -> np.ndarray:
        return self.vetores[[self.posicao[rid] for rid in ids]]

    def salvar(self, pasta: Path, nome: str) -> None:
        pasta.mkdir(parents=True, exist_ok=True)
        np.save(pasta / f"vetores_{nome}.npy", self.vetores)
        (pasta / f"vetores_{nome}_ids.json").write_text(json.dumps(self.ids))

    @staticmethod
    def carregar(pasta: Path, nome: str) -> "IndiceVetorial":
        ids = json.loads((pasta / f"vetores_{nome}_ids.json").read_text())
        return IndiceVetorial(ids, np.load(pasta / f"vetores_{nome}.npy"))
