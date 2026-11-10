"""Métricas de recuperação calculadas sobre os julgamentos de relevância.

O recall é relativo ao pool: o denominador são as avaliações marcadas como
relevantes entre as encontradas por BM25, Titan e modelo local (top-20 de cada).
"""


def recall_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return float("nan")
    return len(set(ranking[:k]) & relevantes) / len(relevantes)


def recall_normalizado_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    """Recall com teto: acertos / min(k, nº de relevantes).

    Quando uma pergunta tem 40 avaliações relevantes, o top-10 nunca passa de 25% de
    recall comum, mesmo perfeito. Dividindo por min(k, relevantes), um top-10 só com
    relevantes vale 1,0, o que permite comparar perguntas com tamanhos diferentes.
    """
    if not relevantes:
        return float("nan")
    return len(set(ranking[:k]) & relevantes) / min(k, len(relevantes))


def cobertura_julgada_at_k(ranking: list[str], julgados: set[str], k: int) -> float:
    """Fração do top-k que foi julgada. Abaixo de 1, as métricas desta configuração
    ficam subestimadas, porque o que está fora do pool conta como não relevante."""
    top = ranking[:k]
    return float("nan") if not top else sum(r in julgados for r in top) / len(top)


def precisao_at_k(ranking: list[str], relevantes: set[str], k: int, julgados: set[str]) -> float:
    """Precisão só entre os itens julgados (itens fora do pool não contam nem a favor nem contra)."""
    top = [r for r in ranking[:k] if r in julgados]
    return float("nan") if not top else sum(r in relevantes for r in top) / len(top)


def mrr(ranking: list[str], relevantes: set[str]) -> float:
    for pos, rid in enumerate(ranking, start=1):
        if rid in relevantes:
            return 1.0 / pos
    return 0.0


def hit_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    return float(bool(set(ranking[:k]) & relevantes))
