"""Seleção diversa das evidências com MMR (Maximal Marginal Relevance).

Avaliações curtas se repetem muito ("Atraso na entrega", "Entrega atrasada").
Escolher só pelo score do rerank enche o contexto de frases quase iguais, e a
resposta fica rasa. O MMR escolhe uma por vez, equilibrando relevância (score do
rerank) e novidade (distância para as já escolhidas, nos vetores do Titan).
"""
import numpy as np


def mmr(ids: list[str], scores: list[float], vetores: np.ndarray, n: int, lambda_: float = 0.7) -> list[int]:
    """Devolve os índices escolhidos. lambda_=1 é só relevância; 0 é só diversidade."""
    if not ids:
        return []
    s = np.asarray(scores, dtype=float)
    rel = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones_like(s)
    v = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)
    sim = v @ v.T
    escolhidos = [int(np.argmax(rel))]
    while len(escolhidos) < min(n, len(ids)):
        restantes = [i for i in range(len(ids)) if i not in escolhidos]
        redundancia = sim[np.ix_(restantes, escolhidos)].max(axis=1)
        valor = lambda_ * rel[restantes] - (1 - lambda_) * redundancia
        escolhidos.append(restantes[int(np.argmax(valor))])
    return escolhidos
