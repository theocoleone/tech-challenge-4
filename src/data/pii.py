"""Mascaramento de dados pessoais (LGPD) nos textos das avaliações.

Aplicado antes de indexar: o que não é necessário para a análise não entra
no índice, no contexto do LLM nem nas respostas.
"""
import re

PADROES = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("[CPF]", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("[CEP]", re.compile(r"\b\d{5}-\d{3}\b")),
    # Com DDD: (11) 98765-4321, 11987654321, 34 99978 4951. Sem DDD, só com hífen
    # (9876-5432), para não confundir com anos ("2017 2018") ou números de pedido.
    ("[TELEFONE]", re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b|\b9?\d{4}-\d{4}\b")),
]


def mascarar_pii(texto: str) -> tuple[str, dict[str, int]]:
    """Substitui PII por marcadores. Retorna o texto e a contagem por tipo."""
    contagem = {}
    for marcador, padrao in PADROES:
        texto, n = padrao.subn(marcador, texto)
        if n:
            contagem[marcador] = n
    return texto, contagem
