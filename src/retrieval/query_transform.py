"""Análise da pergunta: escopo, tipo, busca reescrita e filtros de metadados."""
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

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


def analisar(pergunta: str) -> AnaliseDaPergunta:
    return _cadeia().invoke({"pergunta": pergunta})
