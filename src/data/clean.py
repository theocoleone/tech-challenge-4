"""Limpeza leve e idempotente dos textos das avaliações.

Duas versões de cada texto:
- texto_original: título + mensagem como o cliente escreveu (só PII mascarada).
  É o que aparece citado como evidência.
- texto_normalizado: base para embeddings e BM25. Normaliza Unicode, quebras de
  linha e espaços, separa a pontuação e expande abreviações comuns.

Deliberadamente NÃO faz stemming, NÃO remove stop words, NÃO tira acentos e
NÃO mexe no "não": embeddings e LLM entendem a frase inteira, e a negação
muda o sentido da avaliação ("não chegou").
"""
import math
import re
import unicodedata

ABREVIACOES = {
    "vc": "você",
    "vcs": "vocês",
    "pq": "porque",
    "q": "que",
    "n": "não",
    "nao": "não",
    "tb": "também",
    "tbm": "também",
    "msm": "mesmo",
    "td": "tudo",
    "mt": "muito",
    "mto": "muito",
    "obg": "obrigado",
    "pra": "para",
}

_PONTUACAO = re.compile(r"([.,!?;:()\"])")
_PONTUACAO_REPETIDA = re.compile(r"([.,!?;:])(\s*\1)+")
_PALAVRA = re.compile(r"\w", re.UNICODE)
_ESPACOS = re.compile(r"\s+")
_TEM_LETRA = re.compile(r"[^\W\d_]")


def _texto(valor) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    return str(valor).strip()


def juntar_titulo_mensagem(titulo, mensagem) -> str:
    """Título e mensagem num texto só. Só um deles pode existir."""
    partes = [p for p in (_texto(titulo), _texto(mensagem)) if p]
    if len(partes) == 2 and not partes[0].endswith((".", "!", "?")):
        partes[0] += "."
    return " ".join(partes)


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFC", texto)
    t = _PONTUACAO_REPETIDA.sub(r"\1", t)  # "!!!!" e ". . . ." viram um só
    t = _PONTUACAO.sub(r" \1 ", t)  # antes das abreviações, senão "tbm!!!" não casa
    t = " ".join(ABREVIACOES.get(p.lower(), p) for p in t.split())
    return _ESPACOS.sub(" ", t).strip()


def contar_palavras(texto: str) -> int:
    """Conta só tokens com letra ou dígito (a pontuação separada não conta)."""
    return sum(1 for p in texto.split() if _PALAVRA.search(p))


def informativo(texto: str) -> bool:
    """Falso para texto vazio, só pontuação/emoji/números, ou com menos de 2 caracteres."""
    return len(texto) >= 2 and bool(_TEM_LETRA.search(texto))
