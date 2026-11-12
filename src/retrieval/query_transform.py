"""Análise da pergunta: escopo, tipo, busca reescrita e filtros de metadados.

O LLM faz a análise, e duas regras determinísticas corrigem os erros mais comuns
dele, que apareceram na avaliação do golden set:
- filtro de categoria que a pergunta não citou (ex.: "drones" virava "eletronicos");
- perguntas sem assunto próprio, em que o recorte vem só dos filtros.
"""
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from unidecode import unidecode

from src.config import get_llm
from src.generation.schemas import AnaliseDaPergunta

PROMPTS = Path(__file__).resolve().parent.parent / "generation" / "prompts"


@lru_cache
def _cadeia():
    prompt = ChatPromptTemplate.from_messages([
        ("system", (PROMPTS / "analise_v1.md").read_text()),
        ("human", "{pergunta}"),
    ])
    return prompt | get_llm(estruturado=AnaliseDaPergunta)


# Palavras que não definem assunto: sozinhas, a busca é só "reclamações" ou "elogios" em geral.
GENERICAS = set("""problema problemas reclamacao reclamacoes reclamam reclamaram reclamar reclame
elogio elogios elogiam elogiaram elogiar comentario comentarios comentam dizem relatam relatos
avaliacao avaliacoes cliente clientes compra compras pedido pedidos nota notas loja gostei recomendo
otimo bom ruim pessimo de da do das dos em no na nos nas e o a os as com para por sobre que""".split())


def _tokens(texto: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", unidecode(texto.lower())))


def corrigir_categoria(pergunta: str, analise: AnaliseDaPergunta) -> AnaliseDaPergunta:
    """Só mantém o filtro de categoria se o nome dela aparecer na pergunta."""
    cat = analise.filtros.categoria_pt
    if cat and not (set(cat.split("_")) - {"e", "de"}) & _tokens(pergunta):
        analise.filtros.categoria_pt = None
    return analise


def tem_assunto_proprio(analise: AnaliseDaPergunta) -> bool:
    """Falso quando a busca reescrita, sem palavras genéricas e sem o que já virou
    filtro (categoria), não tem assunto: "problema reclamação cama mesa banho"."""
    sobra = _tokens(analise.consulta_reescrita) - GENERICAS
    if analise.filtros.categoria_pt:
        sobra -= set(analise.filtros.categoria_pt.split("_"))
    return bool(sobra)


def analisar(pergunta: str) -> AnaliseDaPergunta:
    return corrigir_categoria(pergunta, _cadeia().invoke({"pergunta": pergunta}))
