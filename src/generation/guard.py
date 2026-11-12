"""Validação da resposta antes de devolvê-la ao usuário.

1. Citações: todo review_id citado precisa estar entre as avaliações enviadas ao
   modelo. Insight com citação inventada é removido.
2. Linguagem de frequência: a resposta de uma pergunta específica vem de uma
   amostra recuperada por busca, então não pode dizer "a maioria" nem dar
   percentuais. Quando isso aparece, a resposta é refeita uma vez com o aviso.
3. IDs no texto: os códigos das avaliações vão nos insights, não na resposta.
"""
import re
from dataclasses import dataclass

FREQUENCIA = re.compile(
    r"\b(a maioria|maior parte|grande parte|quase todos|todos os clientes|"
    r"a minoria|frequentemente|na maioria|predomin\w*)\b|\d+\s?%", re.IGNORECASE)
ID_AVALIACAO = re.compile(r"\(?\b(?:ID:?\s*)?[0-9a-f]{32}\b\)?,?", re.IGNORECASE)


@dataclass
class Validacao:
    insights: list[dict]
    citacoes_invalidas: int
    linguagem_de_frequencia: list[str]


def validar(resposta: str, insights: list[dict], ids_contexto: set[str]) -> Validacao:
    validos, invalidas = [], 0
    for ins in insights:
        ids = [rid for rid in ins.get("review_ids", []) if rid in ids_contexto]
        invalidas += len(ins.get("review_ids", [])) - len(ids)
        if ids:
            validos.append({**ins, "review_ids": ids})
    achados = [m.group(0) for m in FREQUENCIA.finditer(resposta)]
    return Validacao(validos, invalidas, achados)


def limpar_ids(texto: str) -> str:
    sem_ids = ID_AVALIACAO.sub("", texto)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", sem_ids)).strip()
