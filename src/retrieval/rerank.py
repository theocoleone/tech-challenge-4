"""Reordenação (re-ranking) dos candidatos com um cross-encoder.

Embeddings comparam pergunta e avaliação separadamente (cada uma vira um vetor).
O cross-encoder lê os dois textos JUNTOS e dá uma nota de relevância, o que é mais
preciso, mas também mais caro. Por isso ele só reordena os ~30 candidatos que a
busca híbrida já trouxe, em vez de olhar a base inteira.

O Cohere Rerank do Bedrock não está liberado na conta, então usamos um
cross-encoder multilíngue local. Testamos dois: o mmarco-mMiniLMv2 errou casos
simples em português (pôs "chegou antes do prazo" acima de "demorou 20 dias"
numa pergunta sobre atraso) e o bge-reranker-v2-m3 acertou todos, então ficou ele.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

MODELO_RERANK = "BAAI/bge-reranker-v2-m3"


@lru_cache
def _modelo() -> CrossEncoder:
    return CrossEncoder(MODELO_RERANK, max_length=256)


def reordenar(pergunta: str, candidatos: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """candidatos = [(review_id, texto)]. Devolve [(review_id, score)] do mais ao menos relevante."""
    if not candidatos:
        return []
    scores = _modelo().predict([(pergunta, texto) for _, texto in candidatos])
    return sorted(((rid, float(s)) for (rid, _), s in zip(candidatos, scores)), key=lambda x: x[1], reverse=True)
